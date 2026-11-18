using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Trecs.Collections;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace Trecs.Internal
{
    /// <summary>
    /// Paged-slab allocator with bucketed size classes, used as the shared backing for
    /// persistent <see cref="NativeUniquePtr{T}"/> and <see cref="TrecsList{T}"/>.
    /// Per-frame input-allocated unmanaged data lives in
    /// <see cref="InputNativeUniqueHeap"/>, not in this chunk store.
    ///
    /// <para>
    /// Allocations are bucketed by power-of-two size. Each bucket owns a list of pages,
    /// each page is split into fixed-size slots. Slot reuse is generation-protected via
    /// the upper byte of <see cref="PtrHandle"/>: stale handles to a freed slot are detected
    /// at <see cref="ResolveEntry"/> time. Allocations larger than <see cref="MaxBucketSlotSize"/>
    /// fall through to a dedicated single-slot "huge" page sized for that one allocation.
    /// </para>
    ///
    /// <para>
    /// Allocations and frees are both synchronous. <see cref="Alloc"/> publishes the new
    /// entry to the side table before returning the handle. <see cref="Free"/> calls
    /// <c>AtomicSafetyHandle.CheckDeallocateAndThrow</c> first — if any Burst job is
    /// still using the handle (has it scheduled as a read or write dependency), the call
    /// throws in editor / dev builds; in release without safety checks it's undefined
    /// behaviour, same as <c>NativeList&lt;T&gt;.Dispose()</c>. Callers must complete any
    /// jobs using the handle before freeing it.
    /// </para>
    ///
    /// <para><b>Growth in jobs.</b> A job grows an allocation into a <see cref="DetachedBlock"/>
    /// outside the pages; <see cref="AdoptDetached"/> moves it back at the next phase
    /// boundary.</para>
    ///
    /// <para><b>Threading invariant.</b> Every mutating API is main-thread only.
    /// Concurrency with Burst jobs that resolve via <see cref="NativeHeapResolver"/>
    /// splits into two tiers:
    /// </para>
    /// <list type="bullet">
    ///   <item><b>Execute-phase-safe (concurrent with resolving jobs that do not depend
    ///     on the allocation being mutated):</b>
    ///     <see cref="Alloc"/>, <see cref="AllocExternal"/>, <see cref="Free"/>,
    ///     main-thread <see cref="ResolveEntry"/>.
    ///     <para>Alloc writes side-table slot <c>InUse: 0 → 1</c>, but the slot is fresh
    ///     from the freelist — no concurrent reader holds a handle for it at the moment
    ///     of the write, because the handle being returned is the first to encode this
    ///     slot at this generation. Any subsequent <c>Schedule()</c> on a job that uses
    ///     the handle provides the memory ordering needed to publish it.</para>
    ///     <para>Free flips a slot <c>InUse: 1 → 0</c>, but only after
    ///     <c>CheckDeallocateAndThrow</c> verifies no scheduled job holds the safety
    ///     handle, so a concurrent resolver cannot be in the middle of reading the slot
    ///     it's about to clear. Other jobs scheduled in parallel that resolve unrelated
    ///     handles are unaffected.</para>
    ///     <para>New chunk publication uses a memory barrier (<c>AllocateChunk</c>) so
    ///     the chunk's zero-init is visible before the directory pointer.</para>
    ///   </item>
    ///   <item><b>Submit-time only (no jobs may be running):</b>
    ///     <see cref="ReclaimEmptyPages"/>, <see cref="Serialize"/>,
    ///     <see cref="Deserialize"/>, <see cref="Dispose"/>. These free backing memory
    ///     or rebuild internal state in a transient way that would tear concurrent
    ///     reads. Trecs's standard step model satisfies this by blocking on outstanding
    ///     jobs before each <c>Submit</c> call.
    ///   </item>
    /// </list>
    /// </summary>
    internal sealed class NativeHeap : IDisposable
    {
        static readonly TrecsProfilerMarker _allocMarker = new("NativeHeap.Alloc");
        static readonly TrecsProfilerMarker _freeMarker = new("NativeHeap.Free");
        static readonly TrecsProfilerMarker _adoptDetachedMarker = new("NativeHeap.AdoptDetached");

        readonly TrecsLog _log;

        // Slot-size ladder: powers of two from 16 B up to 64 KB. Above this we use the
        // huge-alloc path (one dedicated page per allocation).
        internal const int MinBucketSlotSize = 16;
        internal const int MaxBucketSlotSize = 64 * 1024;
        internal const int TargetSlotsPerPage = 64;
        internal const int MinPageSize = 4 * 1024; // never allocate a page smaller than this
        const int BucketCount = 13; // MinBucketSlotSize << i up to MaxBucketSlotSize

        PageStore _store;

        // Side-table slot recycling. Slot 0 is reserved as the "null" slot — never returned.
        readonly Stack<int> _freeSideTableSlots = new();
        int _nextFreshSideTableSlot = 1;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        // Tracks addresses currently registered via AllocExternal so we can catch a caller
        // that takes ownership of the same pointer twice (which would leave the chunk store
        // with two slots aliasing the same memory; freeing either leaves the other dangling).
        // Removed when the slot is released. Debug-only — release builds skip the check.
        readonly HashSet<IntPtr> _externalAddresses = new();
#endif

#if DEBUG
        // (handle.Value → call site that allocated it). Populated by Alloc / AllocExternal
        // via [CallerFilePath] / [CallerLineNumber] attributes baked into the IL at user
        // call sites; consumed by the Dispose-time leak warning and desync-friendly
        // serialization for actionable attribution.
        readonly Dictionary<uint, (string File, int Line)> _callerInfoByHandle = new();
#endif

        // Chunked side table: a fixed-capacity directory of pointers to fixed-size
        // chunks of NativeHeapEntry. Appending a new chunk does not move existing
        // chunks, so jobs reading via the resolver can keep resolving handles concurrent
        // with main-thread Allocs that materialise additional slots.
        NativeArray<IntPtr> _chunkDirectory;
        readonly unsafe NativeHeapDetachLog* _detachLog;
        int _sideTableLength;
        NativeHeapResolver _resolver;
        bool _isDisposed;
        int _liveCount;

        // Dense serialization lists: maintained incrementally by Alloc/Free so
        // Serialize can blit two contiguous arrays instead of scanning the
        // entire chunked side table for live entries.
        NativeList<NativeHeapEntryPayload> _densePayloads;
        NativeList<int> _denseSlotIndices;
        NativeList<int> _sparseToDense;

        public unsafe NativeHeap(TrecsLog log)
        {
            _log = log;
            _store = PageStore.Create();

            // Fixed-capacity directory: 16K pointers = 128 KB upfront, covering the
            // entire 24-bit index space. The directory itself never moves; chunks
            // are allocated lazily as slot indices cross chunk boundaries.
            _chunkDirectory = new NativeArray<IntPtr>(
                NativeHeapResolver.MaxChunkCount,
                Allocator.Persistent,
                NativeArrayOptions.ClearMemory
            );

            _densePayloads = new NativeList<NativeHeapEntryPayload>(64, Allocator.Persistent);
            _denseSlotIndices = new NativeList<int>(64, Allocator.Persistent);
            _sparseToDense = new NativeList<int>(1, Allocator.Persistent);

            // Materialise slot 0 as the reserved "null" slot. ClearMemory + the
            // chunk allocator's MemClear guarantee _sideTable[0] reads as default.
            // EnsureSideTableLength also grows _sparseToDense, so dense lists must
            // be allocated first.
            EnsureSideTableLength(1);

            _detachLog = (NativeHeapDetachLog*)
                UnsafeUtility.MallocTracked(
                    sizeof(NativeHeapDetachLog),
                    UnsafeUtility.AlignOf<long>(),
                    Allocator.Persistent,
                    0
                );
            UnsafeUtility.MemClear(_detachLog, sizeof(NativeHeapDetachLog));
            DetachedBlock.InitArenas(_detachLog);

            _resolver = new NativeHeapResolver(_chunkDirectory, _detachLog);
        }

        public int NumLiveAllocations
        {
            get
            {
                TrecsDebugAssert.That(!_isDisposed);
                return _liveCount;
            }
        }

        public int NumPages
        {
            get
            {
                TrecsDebugAssert.That(!_isDisposed);
                return _store.Pages.Length - _store.FreePageIds.Length;
            }
        }

        /// <summary>
        /// Snapshot of internal allocator state — per-bucket page/slot counts, side-table
        /// density, total reserved bytes. Intended for benchmarking and diagnostics only;
        /// kept <c>internal</c> so it isn't part of the public API surface yet.
        /// </summary>
        internal unsafe HeapStatistics GetStatistics()
        {
            TrecsDebugAssert.That(!_isDisposed);
            TrecsDebugAssert.That(
                UnityThreadHelper.IsMainThread,
                "NativeHeap.GetStatistics must be called from the main thread"
            );

            var bucketStats = new BucketStatistics[BucketCount];
            long totalReserved = 0;

            for (int b = 0; b < BucketCount; b++)
            {
                var bucket = _store.Buckets[b];
                int pageCount = 0;
                int liveSlotCount = 0;
                int freeSlotCount = 0;
                long reservedBytes = 0;
                foreach (var pageId in bucket.PageIds)
                {
                    var page = _store.Pages[pageId];
                    if (page.Address == IntPtr.Zero)
                    {
                        continue;
                    }
                    pageCount++;
                    liveSlotCount += page.SlotCount - page.FreeCount;
                    freeSlotCount += page.FreeCount;
                    reservedBytes += (long)page.SlotSize * page.SlotCount;
                }
                bucketStats[b] = new BucketStatistics
                {
                    SlotSize = bucket.SlotSize,
                    SlotsPerPage = bucket.SlotsPerPage,
                    PageCount = pageCount,
                    LiveSlotCount = liveSlotCount,
                    FreeSlotCount = freeSlotCount,
                    ReservedBytes = reservedBytes,
                };
                totalReserved += reservedBytes;
            }

            int numHuge = 0;
            long hugeReserved = 0;
            for (int pageId = 0; pageId < _store.Pages.Length; pageId++)
            {
                var page = _store.Pages[pageId];
                if (page.Address == IntPtr.Zero || page.BucketIdx >= 0)
                {
                    continue;
                }
                numHuge++;
                hugeReserved += (long)page.SlotSize * page.SlotCount;
            }
            totalReserved += hugeReserved;

            return new HeapStatistics
            {
                LiveAllocations = _liveCount,
                NumPages = _store.Pages.Length - _store.FreePageIds.Length,
                NumHugePages = numHuge,
                SideTableLength = _sideTableLength,
                FreeSideTableSlots = _freeSideTableSlots.Count,
                NextFreshSideTableSlot = _nextFreshSideTableSlot,
                HugePagesReservedBytes = hugeReserved,
                TotalReservedBytes = totalReserved,
                Buckets = bucketStats,
            };
        }

        /// <summary>
        /// Per-page occupancy snapshot — one <see cref="PageOccupancyHeader"/> per
        /// currently-allocated page, plus a flat bool span recording per-slot in-use
        /// state. Intended for the heap-stress HUD's fragmentation visualisation;
        /// not part of the public Trecs API surface.
        ///
        /// <para>Caller owns the output lists and is expected to reuse them across
        /// calls. Each call clears them and refills. Marking live slots iterates the
        /// dense list (O(liveCount)); building the page headers is O(numPages).</para>
        /// </summary>
        internal void SnapshotPageOccupancy(
            List<PageOccupancyHeader> outPages,
            List<bool> outLiveSlots
        )
        {
            TrecsDebugAssert.That(!_isDisposed);
            TrecsDebugAssert.That(
                UnityThreadHelper.IsMainThread,
                "NativeHeap.SnapshotPageOccupancy must be called from the main thread"
            );
            TrecsDebugAssert.That(outPages != null && outLiveSlots != null);

            outPages.Clear();
            outLiveSlots.Clear();

            // Pass 1: reserve a header + per-slot bool block for each live page.
            // pageIdToHeaderIdx is keyed by raw pageId (which may have gaps due to
            // reclaimed pages) and maps into the dense outPages list.
            var pageIdToHeaderIdx = new int[_store.Pages.Length];
            for (int pageId = 0; pageId < _store.Pages.Length; pageId++)
            {
                var page = _store.Pages[pageId];
                if (page.Address == IntPtr.Zero)
                {
                    pageIdToHeaderIdx[pageId] = -1;
                    continue;
                }
                pageIdToHeaderIdx[pageId] = outPages.Count;
                outPages.Add(
                    new PageOccupancyHeader
                    {
                        PageId = pageId,
                        BucketIdx = page.BucketIdx,
                        SlotSize = page.SlotSize,
                        SlotCount = page.SlotCount,
                        FreeCount = page.FreeCount,
                        LiveSlotsStartIndex = outLiveSlots.Count,
                    }
                );
                for (int s = 0; s < page.SlotCount; s++)
                {
                    outLiveSlots.Add(false);
                }
            }

            // Pass 2: mark each live entry's slot via the dense list (O(liveCount)
            // instead of O(sideTableLength)).
            for (int n = 0; n < _densePayloads.Length; n++)
            {
                var payload = _densePayloads[n];
                var headerIdx = pageIdToHeaderIdx[payload.PageId];
                TrecsDebugAssert.That(
                    headerIdx >= 0,
                    "SnapshotPageOccupancy: live entry references reclaimed page {0}",
                    payload.PageId
                );
                var header = outPages[headerIdx];
                outLiveSlots[header.LiveSlotsStartIndex + payload.SlotIndex] = true;
            }
        }

        public ref NativeHeapResolver Resolver
        {
            get
            {
                TrecsDebugAssert.That(!_isDisposed);
                return ref _resolver;
            }
        }

        /// <summary>
        /// Allocate a slot at least <paramref name="size"/> bytes wide, aligned to at least
        /// <paramref name="alignment"/>. Returns a handle whose validity is checked against
        /// the side table's generation/in-use bits on every resolve.
        ///
        /// <para>The new entry is written to the side table before the handle is returned,
        /// so any Burst job scheduled afterwards with this handle can resolve it immediately.</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public PtrHandle Alloc(
            int size,
            int alignment,
            int typeHash,
            [CallerFilePath] string callerFile = null,
            [CallerLineNumber] int callerLine = 0
        )
        {
            return Alloc(size, alignment, typeHash, out _, callerFile, callerLine);
        }

        /// <summary>
        /// Same as <see cref="Alloc(int, int, int, string, int)"/>, but additionally returns
        /// the slot's address in <paramref name="address"/>. Callers that need to write the
        /// initial value into the allocation should use this overload to avoid an immediate
        /// follow-up <see cref="ResolveEntry"/>.
        ///
        /// <para><paramref name="callerFile"/> / <paramref name="callerLine"/> are filled by
        /// the compiler at the call site (<c>[CallerFilePath]</c> / <c>[CallerLineNumber]</c>);
        /// pass them through explicitly when forwarding from a wrapper so the recorded site
        /// is the user's, not the wrapper's. Stored under <c>ENABLE_UNITY_COLLECTIONS_CHECKS</c>
        /// only; the leak warning at <see cref="Dispose"/> renders them to make the warning
        /// actionable.</para>
        /// </summary>
        public unsafe PtrHandle Alloc(
            int size,
            int alignment,
            int typeHash,
            out IntPtr address,
            [CallerFilePath] string callerFile = null,
            [CallerLineNumber] int callerLine = 0
        )
        {
            using var _ = _allocMarker.Auto();
            TrecsDebugAssert.That(!_isDisposed);
            TrecsDebugAssert.That(size > 0, "Allocation size must be positive");
            TrecsDebugAssert.That(
                alignment > 0 && (alignment & (alignment - 1)) == 0,
                "Alignment {0} must be a positive power of two",
                alignment
            );
            TrecsDebugAssert.That(
                UnityThreadHelper.IsMainThread,
                "NativeHeap.Alloc must be called from the main thread"
            );

            AllocSlot(
                size,
                alignment,
                out var pageId,
                out var slotIdx,
                out var bucketIdx,
                out var ownsWholePage,
                out var slotByteSize,
                out address
            );

            // Zero the entire slot — not just the caller's requested `size` — so any tail
            // bytes (between size and slotByteSize, or past Count*ElementSize for
            // variable-size containers) are deterministic. Snapshots dump full slot bytes;
            // without this, recycled slots leak the previous tenant's contents into the
            // snapshot non-deterministically across worlds.
            UnsafeUtility.MemClear(address.ToPointer(), slotByteSize);

            var sideTableIdx = AcquireSideTableSlot();

            // The slot is in InUse=0 from the freelist; we own it until we write
            // InUse=1 below. AcquireSideTableSlot guarantees the slot exists, so the
            // generation read here is the slot's true last-used generation. Skip 0 on
            // wrap since 0 means "never allocated."
            var prior = GetEntry(sideTableIdx);
            var nextGen = (byte)(prior.Generation + 1);
            if (nextGen == 0)
                nextGen = 1;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
            var safety = AtomicSafetyHandle.Create();
            AtomicSafetyHandle.SetBumpSecondaryVersionOnScheduleWrite(safety, true);
#endif

            var newEntry = new NativeHeapEntry
            {
                Address = address,
                TypeHash = typeHash,
                Generation = nextGen,
                InUse = 1,
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Safety = safety,
#endif
            };

            var handleValue = NativeHeapResolver.EncodeHandleValue(nextGen, (uint)sideTableIdx);
            // Write the entry to the side table directly. Safe concurrent with running
            // jobs because the slot was free (InUse=0) and no other reader holds a handle
            // for it: the handle we're about to return is the first one to exist for
            // this slot at this generation. Schedule() on any subsequent job provides
            // the ordering needed to publish the entry to that job.
            SetEntry(sideTableIdx, newEntry);
            _liveCount++;

            _sparseToDense[sideTableIdx] = _densePayloads.Length;
            _denseSlotIndices.Add(sideTableIdx);
            _densePayloads.Add(
                new NativeHeapEntryPayload
                {
                    PageId = pageId,
                    SlotIndex = (ushort)slotIdx,
                    OwnsWholePage = (byte)(ownsWholePage ? 1 : 0),
                }
            );

#if DEBUG
            if (callerFile != null)
            {
                _callerInfoByHandle[handleValue] = (callerFile, callerLine);
            }
#endif

            _log.Trace(
                "Alloc: handle={0} bucket={1} page={2} slot={3} addr={4}",
                handleValue,
                bucketIdx,
                pageId,
                slotIdx,
                address.ToInt64()
            );

            return new PtrHandle(handleValue);
        }

        void AllocSlot(
            int size,
            int alignment,
            out int pageId,
            out int slotIdx,
            out int bucketIdx,
            out bool ownsWholePage,
            out int slotByteSize,
            out IntPtr address
        ) =>
            _store.AllocSlot(
                size,
                alignment,
                out pageId,
                out slotIdx,
                out bucketIdx,
                out ownsWholePage,
                out slotByteSize,
                out address
            );

        /// <summary>
        /// Moves a live allocation to a new slot of <paramref name="newSize"/> bytes, keeping
        /// its handle and safety handle. Copies the first <paramref name="copyBytes"/> bytes
        /// and zeroes the rest. Main thread; no job may hold the allocation.
        /// </summary>
        public unsafe void* Reallocate(PtrHandle handle, int newSize, int alignment, int copyBytes)
        {
            TrecsDebugAssert.That(!_isDisposed);
            TrecsDebugAssert.That(
                UnityThreadHelper.IsMainThread,
                "NativeHeap.Reallocate must be called from the main thread"
            );
            NativeHeapResolver.DecodeHandle(handle, out var idxU, out var gen);
            var idx = (int)idxU;
            var slot = GetEntryPtr(idx);
            TrecsAssert.That(
                slot->InUse == 1 && slot->Generation == gen,
                "Reallocate: handle {0} is freed or stale",
                handle.Value
            );
            var payload =
                (NativeHeapEntryPayload*)_densePayloads.GetUnsafePtr() + _sparseToDense[idx];
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckDeallocateAndThrow(slot->Safety);
            TrecsDebugAssert.That(
                payload->OwnsWholePage == 0
                    || !_externalAddresses.Contains(_store.Pages[payload->PageId].Address),
                "Reallocate: handle {0} owns an external pointer",
                handle.Value
            );
#endif
            return MoveToNewSlot(ref _store, slot, payload, newSize, alignment, copyBytes);
        }

        static unsafe void* MoveToNewSlot(
            ref PageStore store,
            NativeHeapEntry* slot,
            NativeHeapEntryPayload* payload,
            int newSize,
            int alignment,
            int copyBytes
        )
        {
            TrecsDebugAssert.That(
                copyBytes >= 0 && copyBytes <= newSize,
                "MoveToNewSlot: copyBytes {0} outside [0, {1}]",
                copyBytes,
                newSize
            );
            store.AllocSlot(
                newSize,
                alignment,
                out var pageId,
                out var slotIdx,
                out _,
                out var ownsWholePage,
                out var slotByteSize,
                out var address
            );
            var dst = (byte*)address.ToPointer();
            UnsafeUtility.MemCpy(dst, slot->Address.ToPointer(), copyBytes);
            UnsafeUtility.MemClear(dst + copyBytes, slotByteSize - copyBytes);

            store.ReturnSlot(payload->PageId, payload->SlotIndex, payload->OwnsWholePage != 0);
            payload->PageId = pageId;
            payload->SlotIndex = (ushort)slotIdx;
            payload->OwnsWholePage = (byte)(ownsWholePage ? 1 : 0);

            slot->Address = address;
            slot->IsDetached = 0;
            return dst;
        }

        /// <summary>
        /// Moves every detached block (grown by a job since the last call) into a page
        /// slot, in side-table order so placement matches on every peer, then frees the
        /// arenas. Main thread, no job running: call at each phase boundary. No work when
        /// nothing was detached.
        /// </summary>
        internal unsafe void AdoptDetached()
        {
            TrecsDebugAssert.That(!_isDisposed);
            if (_detachLog->HasDetached == 0)
            {
                return;
            }
            using var _ = _adoptDetachedMarker.Auto();
            new AdoptDetachedJob
            {
                Store = _store,
                Log = _detachLog,
                ChunkDirectory = (IntPtr*)_chunkDirectory.GetUnsafeReadOnlyPtr(),
                ChunkCount =
                    (_sideTableLength + NativeHeapResolver.ChunkSize - 1)
                    >> NativeHeapResolver.ChunkSizeBits,
                SparseToDense = (int*)_sparseToDense.GetUnsafeReadOnlyPtr(),
                DensePayloads = (NativeHeapEntryPayload*)_densePayloads.GetUnsafePtr(),
            }.Run();
        }

        [BurstCompile]
        unsafe struct AdoptDetachedJob : IJob
        {
            public PageStore Store;

            [NativeDisableUnsafePtrRestriction]
            public NativeHeapDetachLog* Log;

            [NativeDisableUnsafePtrRestriction]
            public IntPtr* ChunkDirectory;
            public int ChunkCount;

            [NativeDisableUnsafePtrRestriction]
            public int* SparseToDense;

            [NativeDisableUnsafePtrRestriction]
            public NativeHeapEntryPayload* DensePayloads;

            public void Execute()
            {
                Log->HasDetached = 0;
                var wordCount = (ChunkCount + 63) >> 6;
                for (int w = 0; w < wordCount; w++)
                {
                    var chunkBits = Log->ChunkBits[w];
                    Log->ChunkBits[w] = 0;
                    while (chunkBits != 0)
                    {
                        var chunk = (w << 6) + math.tzcnt(chunkBits);
                        chunkBits &= chunkBits - 1;
                        var chunkBase = (NativeHeapEntry*)ChunkDirectory[chunk].ToPointer();
                        var entryBits = (long*)(chunkBase + NativeHeapResolver.ChunkSize);
                        for (int e = 0; e < NativeHeapResolver.ChunkSize / 64; e++)
                        {
                            var bits = entryBits[e];
                            entryBits[e] = 0;
                            while (bits != 0)
                            {
                                var inChunk = (e << 6) + math.tzcnt(bits);
                                bits &= bits - 1;
                                var slot = chunkBase + inChunk;
                                // Freed, or already moved back by Reallocate since.
                                if (slot->InUse == 0 || slot->IsDetached == 0)
                                {
                                    continue;
                                }
                                var idx = (chunk << NativeHeapResolver.ChunkSizeBits) + inChunk;
                                var size = DetachedBlock.GetSize(slot->Address.ToPointer());
                                MoveToNewSlot(
                                    ref Store,
                                    slot,
                                    DensePayloads + SparseToDense[idx],
                                    size,
                                    DetachedBlock.MaxAlignment,
                                    size
                                );
                            }
                        }
                    }
                }
                DetachedBlock.FreeArenaChunks(Log);
            }
        }

        /// <summary>
        /// Releases a previously-allocated handle, synchronously. The side-table slot is
        /// marked <c>InUse=0</c>, the bucket slot (or huge page) is returned, and the
        /// per-allocation safety handle is released — all before <see cref="Free"/> returns.
        ///
        /// <para>If any Burst job currently holds this allocation's safety handle (i.e. has
        /// it scheduled as a read or write dependency),
        /// <c>AtomicSafetyHandle.CheckDeallocateAndThrow</c> throws in editor / dev builds.
        /// Callers must arrange for no job to be using the handle when <c>Free</c> is called
        /// — typically by completing the relevant jobs first, or by calling <c>Free</c> at a
        /// phase-boundary where the framework has already drained outstanding jobs.</para>
        ///
        /// <para>In release builds with <c>ENABLE_UNITY_COLLECTIONS_CHECKS</c> off the
        /// dependency check is compiled out and freeing while a job uses the handle is
        /// undefined behaviour — same contract as <c>NativeList&lt;T&gt;.Dispose()</c>.</para>
        ///
        /// <para>If <paramref name="onDrained"/> is non-null, it runs after the safety check
        /// passes but before the chunk-store slot is released. Use this for any caller-owned
        /// memory tied to this allocation (e.g. the <c>TrecsList</c> data buffer) — the entry
        /// passed in is still pointing at valid slot memory; the chunk-store releases the slot
        /// after the callback returns. Prefer a static method assigned to a static readonly
        /// field to avoid per-call GC allocation from closure capture.</para>
        /// </summary>
        public unsafe void Free(PtrHandle handle)
        {
            using var _ = _freeMarker.Auto();
            TrecsDebugAssert.That(!_isDisposed);
            TrecsDebugAssert.That(!handle.IsNull, "Attempted to free null PtrHandle");
            TrecsDebugAssert.That(
                UnityThreadHelper.IsMainThread,
                "NativeHeap.Free must be called from the main thread"
            );

            NativeHeapResolver.DecodeHandle(handle, out var idxU, out var gen);
            var idx = (int)idxU;
            TrecsDebugAssert.That(
                idx < _sideTableLength,
                "Free: handle {0} side-table index {1} out of range",
                handle.Value,
                idx
            );
            var entry = GetEntry(idx);
            TrecsDebugAssert.That(entry.InUse == 1, "Free: handle {0} already freed", handle.Value);
            TrecsDebugAssert.That(
                entry.Generation == gen,
                "Free: stale handle {0} (slot gen {1})",
                handle.Value,
                entry.Generation
            );

#if ENABLE_UNITY_COLLECTIONS_CHECKS
            // Throws InvalidOperationException if any job has this allocation scheduled
            // as a dependency. Caller is responsible for completing those jobs before
            // calling Free. Mirrors NativeList<T>.Dispose() semantics.
            AtomicSafetyHandle.CheckDeallocateAndThrow(entry.Safety);
            AtomicSafetyHandle.Release(entry.Safety);
#endif

            var denseIdx = _sparseToDense[idx];
            TrecsDebugAssert.That(
                denseIdx >= 0 && denseIdx < _densePayloads.Length,
                "Free: dense index {0} for side-table slot {1} out of range [0, {2})",
                denseIdx,
                idx,
                _densePayloads.Length
            );
            var payload = _densePayloads[denseIdx];
            var lastDense = _densePayloads.Length - 1;
            if (denseIdx != lastDense)
            {
                _densePayloads[denseIdx] = _densePayloads[lastDense];
                _denseSlotIndices[denseIdx] = _denseSlotIndices[lastDense];
                _sparseToDense[_denseSlotIndices[denseIdx]] = denseIdx;
            }
            _densePayloads.Length--;
            _denseSlotIndices.Length--;
            _sparseToDense[idx] = -1;

            entry.InUse = 0;
            entry.IsDetached = 0;
            entry.Address = IntPtr.Zero;
            SetEntry(idx, entry);

            ReturnSlot(payload.PageId, payload.SlotIndex, payload.OwnsWholePage != 0);
            _freeSideTableSlots.Push(idx);
            _liveCount--;
#if DEBUG
            _callerInfoByHandle.Remove(handle.Value);
#endif
            _log.Trace("Free: handle={0}", handle.Value);
        }

        /// <summary>
        /// Main-thread entry resolution. Behaves identically to <see cref="Resolver"/>'s
        /// <c>ResolveEntry</c> — provided as a managed-friendly entry point that doesn't
        /// require capturing the resolver struct, and asserts main-thread.
        /// </summary>
        public NativeHeapEntry ResolveEntry(PtrHandle handle)
        {
            TrecsDebugAssert.That(!_isDisposed);
            TrecsDebugAssert.That(!handle.IsNull, "Attempted to resolve null PtrHandle");
            TrecsDebugAssert.That(
                UnityThreadHelper.IsMainThread,
                "NativeHeap.ResolveEntry is main-thread only; jobs use NativeHeapResolver"
            );

            return _resolver.ResolveEntry(handle);
        }

        internal NativeHeapEntryPayload ResolvePayload(PtrHandle handle)
        {
            TrecsDebugAssert.That(!_isDisposed);
            TrecsDebugAssert.That(!handle.IsNull, "Attempted to resolve null PtrHandle");
            NativeHeapResolver.DecodeHandle(handle, out var idxU, out _);
            var idx = (int)idxU;
            var denseIdx = _sparseToDense[idx];
            TrecsDebugAssert.That(
                denseIdx >= 0 && denseIdx < _densePayloads.Length,
                "ResolvePayload: dense index {0} for side-table slot {1} out of range",
                denseIdx,
                idx
            );
            return _densePayloads[denseIdx];
        }

        /// <summary>
        /// Walks the page list and frees any page whose every slot is unallocated. Returns
        /// the number of pages reclaimed. Submit-time only: no jobs may be resolving handles
        /// while this runs.
        /// </summary>
        internal int ReclaimEmptyPages()
        {
            using var _ = TrecsProfiling.Start("NativeHeap.ReclaimEmptyPages");
            TrecsDebugAssert.That(!_isDisposed);
            TrecsDebugAssert.That(UnityThreadHelper.IsMainThread);

            int reclaimed = 0;
            for (int pageId = 0; pageId < _store.Pages.Length; pageId++)
            {
                var page = _store.Pages[pageId];
                if (page.Address == IntPtr.Zero || page.FreeCount != page.SlotCount)
                    continue;

                _store.ReclaimPage(pageId);
                reclaimed++;
            }
            return reclaimed;
        }

        internal bool SuppressDisposeWarnings { get; set; }

        public unsafe void Dispose()
        {
            TrecsDebugAssert.That(!_isDisposed);

            DetachedBlock.DisposeArenas(_detachLog);
            UnsafeUtility.FreeTracked(_detachLog, Allocator.Persistent);

            // Any side-table slots still in use indicate leaked allocations. Render a
            // single grouped warning (type + call site + occurrence count), then release
            // their safety handles so the safety system's slot table doesn't leak.
            // (Bucket slots themselves are reclaimed by the unconditional FreePage loop
            // below, regardless of leak state.)
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            if (!SuppressDisposeWarnings)
                WarnAboutLiveLeaks();

            for (int n = 0; n < _denseSlotIndices.Length; n++)
            {
                var entry = GetEntry(_denseSlotIndices[n]);
                AtomicSafetyHandle.CheckDeallocateAndThrow(entry.Safety);
                AtomicSafetyHandle.Release(entry.Safety);
            }
#endif
#if DEBUG
            _callerInfoByHandle.Clear();
#endif

            _store.Dispose();
            _freeSideTableSlots.Clear();
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _externalAddresses.Clear();
#endif

            DisposeAllChunks();
            _chunkDirectory.Dispose();
            _densePayloads.Dispose();
            _denseSlotIndices.Dispose();
            _sparseToDense.Dispose();
            _sideTableLength = 0;
            _isDisposed = true;
        }

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        /// <summary>
        /// Iterates the dense live-entry list at <see cref="Dispose"/> time, groups entries
        /// by <c>(TypeHash, caller file, caller line)</c>, and logs one warning per
        /// distinct group with a <c>(×N)</c> suffix. Types are resolved via
        /// <see cref="TypeIdReverseLookup.TryGetTypeFromId"/>; slots whose id isn't
        /// registered there render as <c>"TypeId 0x…"</c>. Slots with no recorded call
        /// site (raw internal allocations that didn't propagate <c>[CallerFilePath]</c>)
        /// render as <c>"&lt;unknown caller&gt;"</c>.
        /// </summary>
        void WarnAboutLiveLeaks()
        {
            if (_liveCount == 0 || !_log.IsWarningEnabled())
            {
                return;
            }

            // Group by (typeHash, file, line) so identical leaks dedupe with a count.
            var groups = new IterableDictionary<(int TypeHash, string File, int Line), int>();
            for (int n = 0; n < _densePayloads.Length; n++)
            {
                var sideTableIdx = _denseSlotIndices[n];
                var entry = GetEntry(sideTableIdx);

                string callerFile = null;
                int callerLine = 0;
#if DEBUG
                var handleValue = NativeHeapResolver.EncodeHandleValue(
                    entry.Generation,
                    (uint)sideTableIdx
                );
                if (_callerInfoByHandle.TryGetValue(handleValue, out var caller))
                {
                    callerFile = caller.File;
                    callerLine = caller.Line;
                }
#endif
                var key = (entry.TypeHash, callerFile, callerLine);
                groups.TryGetValue(key, out var count);
                groups[key] = count + 1;
            }

            _log.Warning("NativeHeap: {0} undisposed allocation(s) at teardown:", _liveCount);
            foreach (var (key, count) in groups)
            {
                var typeName = TypeIdReverseLookup.TryGetTypeFromId(
                    new TypeId(key.TypeHash),
                    out var t
                )
                    ? t.GetPrettyName()
                    : $"TypeId 0x{key.TypeHash:X8}";
                var site = key.File != null ? $"{key.File}:{key.Line}" : "<unknown caller>";
                var suffix = count > 1 ? $" (×{count})" : "";
                _log.Warning("  - {0} at {1}{2}", typeName, site, suffix);
            }
        }
#endif

        // ─── Serialization ───────────────────────────────────────

        // Also covers the block layouts of the containers stored in it (TrecsList, TrecsDictionary).
        const int SerializationVersion = 5;

        /// <summary>
        /// Bulk-dumps the chunk store's full state in a few wide blits. Replaces the
        /// previous model where consuming heaps drove restoration one entry at a time.
        /// Must be called before each heap's <c>Serialize</c> — on load the heaps assume
        /// every chunk-store slot they reference has already been re-materialised.
        ///
        /// <para>The wire format is deterministic: identical chunk-store state produces
        /// identical bytes. Process-local fields (<c>AtomicSafetyHandle</c>, raw page
        /// addresses) are stripped or replaced on load.</para>
        /// </summary>
        public unsafe void Serialize(ISerializationWriter writer)
        {
            using var _ = TrecsProfiling.Start("NativeHeap.Serialize");
            TrecsDebugAssert.That(!_isDisposed);
            TrecsDebugAssert.That(UnityThreadHelper.IsMainThread);
            AdoptDetached();

            if (writer.HasFlag(SerializationFlags.DesyncFriendlyHeaps))
            {
                SerializeDesyncFriendly(writer);
                return;
            }

            var heapHeader = new SerializedHeapHeader
            {
                Version = SerializationVersion,
                LiveCount = _liveCount,
                SideTableLength = _sideTableLength,
                NextFreshSideTableSlot = _nextFreshSideTableSlot,
                NumPages = _store.Pages.Length,
            };
            writer.BlitWriteRawBytes(
                "Header",
                &heapHeader,
                UnsafeUtility.SizeOf<SerializedHeapHeader>()
            );

            using (TrecsProfiling.Start("ZeroFreeSlots"))
            {
                ZeroFreeSlots();
            }

            using (TrecsProfiling.Start("Writing pages"))
            {
                for (int pageId = 0; pageId < _store.Pages.Length; pageId++)
                {
                    var page = _store.Pages[pageId];
                    if (page.Address == IntPtr.Zero)
                    {
                        var nullHeader = new SerializedPageHeader { Kind = (byte)PageKind.Null };
                        writer.BlitWriteRawBytes(
                            "PageHeader",
                            &nullHeader,
                            UnsafeUtility.SizeOf<SerializedPageHeader>()
                        );
                        continue;
                    }

                    var kind = page.BucketIdx >= 0 ? PageKind.Bucket : PageKind.SingleSlot;
                    var pageHeader = new SerializedPageHeader
                    {
                        Kind = (byte)kind,
                        SlotSize = page.SlotSize,
                        SlotCount = page.SlotCount,
                        FreeCount = page.FreeCount,
                        Alignment = page.Alignment,
                        BucketIdx = page.BucketIdx,
                    };
                    writer.BlitWriteRawBytes(
                        "PageHeader",
                        &pageHeader,
                        UnsafeUtility.SizeOf<SerializedPageHeader>()
                    );

                    var pageBytes =
                        page.BucketIdx >= 0 ? page.SlotSize * page.SlotCount : page.SlotSize;
                    writer.BlitWriteRawBytes("Data", page.Address.ToPointer(), pageBytes);
                }
            }

            using (TrecsProfiling.Start("Writing side-table"))
            {
                WriteSideTableAndFreeSlots(writer);
            }

            _log.Trace(
                "Chunk store serialized: pages={0} liveEntries={1} sideTableLength={2}",
                _store.Pages.Length,
                _liveCount,
                _sideTableLength
            );
        }

        unsafe void ZeroFreeSlots()
        {
            for (int b = 0; b < BucketCount; b++)
            {
                foreach (var free in _store.Buckets[b].FreeSlots)
                {
                    var page = _store.Pages[free.x];
                    var addr = (byte*)page.Address.ToPointer() + free.y * page.SlotSize;
                    UnsafeUtility.MemClear(addr, page.SlotSize);
                }
            }
        }

        // One entry per allocation, keyed by side-table slot, so a diff lines
        // up allocation by allocation instead of by page layout. Caller
        // file and line name the entry in DEBUG builds only.
        unsafe void SerializeDesyncFriendly(ISerializationWriter writer)
        {
            writer.Write<int>("LiveCount", _liveCount);

            for (int n = 0; n < _densePayloads.Length; n++)
            {
                var payload = _densePayloads[n];
                var sideTableIdx = _denseSlotIndices[n];
                var entry = GetEntry(sideTableIdx);
                var page = _store.Pages[payload.PageId];
                var slotSize = page.SlotSize;
                var slotAddr = (byte*)page.Address.ToPointer() + payload.SlotIndex * slotSize;

                string scopeName = null;
#if DEBUG
                uint handleValue = (uint)entry.Generation << 24 | (uint)sideTableIdx;
                if (_callerInfoByHandle.TryGetValue(handleValue, out var caller))
                {
                    var fileName = Path.GetFileNameWithoutExtension(caller.File);
                    scopeName = $"{fileName}_L{caller.Line}_Slot{sideTableIdx}";
                }
#endif
                if (scopeName == null)
                {
                    var typeName = TypeIdReverseLookup.TryGetTypeFromId(
                        new TypeId(entry.TypeHash),
                        out var type
                    )
                        ? type.GetPrettyName()
                        : $"Type{entry.TypeHash}";
                    scopeName = $"Slot{sideTableIdx}_{typeName}";
                }
                writer.PushScope(scopeName);
                writer.Write<int>("TypeHash", entry.TypeHash);
                writer.Write<int>("Size", slotSize);
                writer.BlitWriteRawBytes("Data", slotAddr, slotSize);
                writer.PopScope();
            }
        }

        /// <summary>
        /// Restores the chunk store from a stream produced by <see cref="Serialize"/>.
        /// Must be called on a freshly-constructed, empty chunk store (no allocations
        /// since the constructor). Each heap's <c>Deserialize</c> runs after this and
        /// can assume every saved handle resolves to live, restored memory.
        /// </summary>
        public unsafe void Deserialize(ISerializationReader reader)
        {
            using var _ = TrecsProfiling.Start("NativeHeap.Deserialize");
            TrecsDebugAssert.That(!_isDisposed);
            TrecsDebugAssert.That(UnityThreadHelper.IsMainThread);
            // Clears the detach bits; the reset below then frees the adopted slots with the rest.
            AdoptDetached();
            // No "live count must be 0" pre-assert: ResetForDeserialize below wipes
            // everything (pages, side-table generations, bucket bookkeeping, live count).
            // Persistent NativeUniquePtr / TrecsList allocations live entirely here, so
            // there's no upstream bookkeeping to ClearAll first; any leftover live slots
            // from before deserialize get reclaimed by the reset along with their safety
            // handles.

            // Read the header upfront so the reset below can scope its
            // side-table memclear to only the slot range that actually needs
            // clearing — slots that were "fresh" (untouched) under the old
            // _nextFreshSideTableSlot but now fall past the new
            // _nextFreshSideTableSlot. For the rollback shape (save then
            // immediately reload the same world), that range is empty, so the
            // memclear cost drops to zero — the largest single win in this
            // path. See ResetForDeserialize's ZeroChunks loop for the
            // rationale on the partial wipe.
            SerializedHeapHeader heapHeader;
            reader.BlitReadRawBytes(
                "Header",
                &heapHeader,
                UnsafeUtility.SizeOf<SerializedHeapHeader>()
            );
            TrecsAssert.That(
                heapHeader.Version == SerializationVersion,
                "Deserialize: chunk-store snapshot version {0} does not match expected {1}",
                heapHeader.Version,
                SerializationVersion
            );

            var liveCount = heapHeader.LiveCount;
            var sideTableLength = heapHeader.SideTableLength;
            var nextFreshSideTableSlot = heapHeader.NextFreshSideTableSlot;
            var numPages = heapHeader.NumPages;

            // Reset back to a fresh-construction state: free any stale bucket pages,
            // zero-init the side-table slots that need it (see partial-wipe
            // rationale above), reset bucket and free-slot bookkeeping.
            // Required because heap.ClearAll empties live entries but leaves unused
            // bucket pages and generation-bumped slots behind.
            using (TrecsProfiling.Start("ResetForDeserialize"))
            {
                ResetForDeserialize(nextFreshSideTableSlot);
            }

            // Materialise side-table chunks up to the saved length. Chunks are zero-init
            // so unused slots read as default (InUse=0) without any extra work.
            using (TrecsProfiling.Start("EnsureSideTableLength"))
            {
                EnsureSideTableLength(sideTableLength);
            }

            // Pages: allocate fresh persistent memory, blit bytes in, rebuild Page objects.
            using (TrecsProfiling.Start("Reading pages"))
            {
                for (int pageId = 0; pageId < numPages; pageId++)
                {
                    SerializedPageHeader pageHeader;
                    reader.BlitReadRawBytes(
                        "PageHeader",
                        &pageHeader,
                        UnsafeUtility.SizeOf<SerializedPageHeader>()
                    );
                    var kind = (PageKind)pageHeader.Kind;
                    if (kind == PageKind.Null)
                    {
                        _store.Pages.Add(default);
                        continue;
                    }

                    var slotSize = pageHeader.SlotSize;
                    var slotCount = pageHeader.SlotCount;
                    var freeCount = pageHeader.FreeCount;
                    var alignment = pageHeader.Alignment;
                    var bucketIdx = pageHeader.BucketIdx;

                    var pageBytes = bucketIdx >= 0 ? slotSize * slotCount : slotSize;
                    // Total bytes as `size`, items=1: the convention PageStore allocates and
                    // frees pages with.
                    var pagePtr = AllocatorManager.Allocate(
                        Allocator.Persistent,
                        pageBytes,
                        alignment,
                        items: 1
                    );
                    TrecsDebugAssert.That(
                        pagePtr != null,
                        "Deserialize: AllocatorManager returned null for page {0} ({1} bytes)",
                        pageId,
                        pageBytes
                    );
                    reader.BlitReadRawBytes("Data", pagePtr, pageBytes);

                    var page = new Page
                    {
                        Address = new IntPtr(pagePtr),
                        BucketIdx = bucketIdx,
                        SlotSize = slotSize,
                        SlotCount = slotCount,
                        FreeCount = freeCount,
                        Alignment = alignment,
                    };
                    _store.Pages.Add(page);
                    if (bucketIdx >= 0)
                    {
                        _store.Buckets[bucketIdx].PageIds.Add(pageId);
                    }
                }
            }

            // Sparse side-table entries.
            int numLiveEntries;
            reader.BlitReadRawBytes("NumLiveEntries", &numLiveEntries, sizeof(int));
            TrecsDebugAssert.That(
                numLiveEntries == liveCount,
                "Deserialize: numLiveEntries {0} does not match liveCount {1}",
                numLiveEntries,
                liveCount
            );

            using (TrecsProfiling.Start("Reading side-table entries"))
            {
                _denseSlotIndices.ResizeUninitialized(numLiveEntries);
                _densePayloads.ResizeUninitialized(numLiveEntries);

                reader.BlitReadRawBytes(
                    "SlotIndices",
                    _denseSlotIndices.GetUnsafePtr(),
                    numLiveEntries * UnsafeUtility.SizeOf<int>()
                );
                reader.BlitReadRawBytes(
                    "Entries",
                    _densePayloads.GetUnsafePtr(),
                    numLiveEntries * UnsafeUtility.SizeOf<NativeHeapEntryPayload>()
                );
                var typeHashes = new NativeArray<int>(
                    numLiveEntries,
                    Allocator.TempJob,
                    NativeArrayOptions.UninitializedMemory
                );
                var generations = new NativeArray<byte>(
                    numLiveEntries,
                    Allocator.TempJob,
                    NativeArrayOptions.UninitializedMemory
                );
                reader.BlitReadRawBytes(
                    "TypeHashes",
                    typeHashes.GetUnsafePtr(),
                    numLiveEntries * sizeof(int)
                );
                reader.BlitReadRawBytes("Generations", generations.GetUnsafePtr(), numLiveEntries);

                // Grow _sparseToDense to cover the full side-table range, filled with -1.
                if (_sparseToDense.Length < sideTableLength)
                {
                    var oldLen = _sparseToDense.Length;
                    _sparseToDense.Resize(sideTableLength, NativeArrayOptions.UninitializedMemory);
                    var sparsePtr = (int*)_sparseToDense.GetUnsafePtr() + oldLen;
                    UnsafeUtility.MemSet(
                        sparsePtr,
                        0xFF,
                        (long)(sideTableLength - oldLen) * sizeof(int)
                    );
                }

                new DeserializeSideTableJob
                {
                    DenseSlotIndices = _denseSlotIndices,
                    DensePayloads = _densePayloads,
                    TypeHashes = typeHashes,
                    Generations = generations,
                    Pages = _store.Pages,
                    ChunkDirectory = _chunkDirectory,
                    SparseToDense = _sparseToDense,
                }
                    .Schedule(numLiveEntries, JobsUtil.ChooseBatchSize(numLiveEntries))
                    .Complete();
                typeHashes.Dispose();
                generations.Dispose();

#if ENABLE_UNITY_COLLECTIONS_CHECKS
                for (int n = 0; n < numLiveEntries; n++)
                {
                    var slotIdx = _denseSlotIndices[n];
                    TrecsDebugAssert.That(
                        slotIdx > 0 && slotIdx < sideTableLength,
                        "Deserialize: slotIdx {0} out of range [1, {1})",
                        slotIdx,
                        sideTableLength
                    );
                    var entry = GetEntry(slotIdx);
                    var payload = _densePayloads[n];
                    TrecsDebugAssert.That(
                        entry.Address != IntPtr.Zero,
                        "Deserialize: live entry references reclaimed page {0}",
                        payload.PageId
                    );
                    entry.Safety = AtomicSafetyHandle.Create();
                    AtomicSafetyHandle.SetBumpSecondaryVersionOnScheduleWrite(entry.Safety, true);
                    SetEntry(slotIdx, entry);
                }
#endif
            }

            // Free-slot / free-page-id stacks: read in saved order and Push directly,
            // which restores the same bottom-to-top arrangement we wrote. The
            // side-table variant also patches each slot's Generation byte from the saved
            // value (see comment in Serialize).
            using (TrecsProfiling.Start("Reading free-slot stacks"))
            {
                ReadSideTableFreeSlots(reader);
                ReadStack(reader, "freePageIds", ref _store.FreePageIds);
            }

            using (TrecsProfiling.Start("Reading bucket free-slots"))
            {
                for (int b = 0; b < BucketCount; b++)
                {
                    ref var bucket = ref _store.Buckets[b];
                    int count;
                    reader.BlitReadRawBytes("BucketFreeSlotsCount", &count, sizeof(int));
                    var pageIds = new NativeArray<int>(count, Allocator.Temp);
                    var slotIdxs = new NativeArray<int>(count, Allocator.Temp);
                    reader.BlitReadRawBytes(
                        "PageIds",
                        NativeArrayUnsafeUtility.GetUnsafePtr(pageIds),
                        count * UnsafeUtility.SizeOf<int>()
                    );
                    reader.BlitReadRawBytes(
                        "SlotIndices",
                        NativeArrayUnsafeUtility.GetUnsafePtr(slotIdxs),
                        count * UnsafeUtility.SizeOf<int>()
                    );
                    // Data is top-to-bottom; push in reverse so top ends up on top.
                    for (int j = count - 1; j >= 0; j--)
                        bucket.FreeSlots.Add(new int2(pageIds[j], slotIdxs[j]));
                    pageIds.Dispose();
                    slotIdxs.Dispose();
                }
            }

            _nextFreshSideTableSlot = nextFreshSideTableSlot;
            _liveCount = liveCount;

            _log.Debug(
                "Chunk store deserialized: pages={0} liveEntries={1} sideTableLength={2}",
                _store.Pages.Length,
                _liveCount,
                _sideTableLength
            );
        }

        // Wipe everything Deserialize is about to repopulate. Takes the new
        // world's _nextFreshSideTableSlot so the side-table memclear can be
        // scoped to just the slots that need it: the live-vs-free state for
        // slots [1, newNextFreshSlot) is restored by SetEntry (live) and by
        // ReadSideTableFreeSlots (which writes a full default-state entry,
        // not just a Generation patch — so a slot that was live in the old
        // world but free in the new lands in the right state); slots
        // [newNextFreshSlot, oldNextFreshSlot) were "used" in the old world
        // (either live or in the free list) and won't be touched by the new
        // world's deserialize, so they have to be cleared here back to
        // fresh; slots beyond oldNextFreshSlot were never used and are
        // already fresh (AllocateChunk zero-inits new chunks; previous
        // resets left old chunks in the same state). For the rollback shape
        // (newNextFreshSlot == oldNextFreshSlot), the wipe range is empty.
        unsafe void ResetForDeserialize(int newNextFreshSlot)
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            using (TrecsProfiling.Start("SafetyHandleRelease"))
            {
                // Release safety handles for live slots before we wipe them —
                // dropping the side-table chunk's bytes leaves Unity's safety-handle
                // slots dangling otherwise. Iterates the dense list (O(liveCount))
                // rather than scanning the full side table.
                for (int n = 0; n < _denseSlotIndices.Length; n++)
                {
                    var entry = GetEntry(_denseSlotIndices[n]);
                    AtomicSafetyHandle.CheckDeallocateAndThrow(entry.Safety);
                    AtomicSafetyHandle.Release(entry.Safety);
                }
            }
#endif
#if DEBUG
            _callerInfoByHandle.Clear();
#endif

            using (TrecsProfiling.Start("FreePages"))
            {
                // Free pages we still own. With stateless heaps, this also reclaims pages
                // for any handles that weren't explicitly disposed before the snapshot load.
                _store.Clear();
                _freeSideTableSlots.Clear();
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                _externalAddresses.Clear();
#endif
            }

            using (TrecsProfiling.Start("ZeroChunks"))
            {
                // Wipe only the slot range [newNextFreshSlot, oldNextFreshSlot) —
                // slots the old world had touched (live or in the free list)
                // but the new world won't, so they need to look "fresh"
                // ({InUse=0, Generation=0, ...}) post-restore. Slots <
                // newNextFreshSlot are written explicitly by SetEntry or by
                // ReadSideTableFreeSlots; slots >= oldNextFreshSlot are
                // already fresh and need no work. Empty range when both
                // worlds have the same _nextFreshSideTableSlot (the common
                // rollback case), which avoids the bulk per-chunk memclear
                // entirely. We don't try to be cleverer than that — partial-
                // chunk memclears are fine; the case worth optimizing is the
                // rollback one, and it's already zero work above.
                var oldNextFreshSlot = _nextFreshSideTableSlot;
                if (newNextFreshSlot < oldNextFreshSlot)
                {
                    var entrySize = UnsafeUtility.SizeOf<NativeHeapEntry>();
                    var chunkSize = NativeHeapResolver.ChunkSize;
                    var chunkSizeBits = NativeHeapResolver.ChunkSizeBits;
                    int slot = newNextFreshSlot;
                    while (slot < oldNextFreshSlot)
                    {
                        var chunkIdx = slot >> chunkSizeBits;
                        var slotInChunk = slot - (chunkIdx << chunkSizeBits);
                        var slotsLeftInChunk = chunkSize - slotInChunk;
                        var slotsToClear = Math.Min(slotsLeftInChunk, oldNextFreshSlot - slot);
                        var ptr = _chunkDirectory[chunkIdx];
                        if (ptr != IntPtr.Zero)
                        {
                            UnsafeUtility.MemClear(
                                (byte*)ptr.ToPointer() + (long)slotInChunk * entrySize,
                                (long)entrySize * slotsToClear
                            );
                        }
                        slot += slotsToClear;
                    }
                }
            }

            _densePayloads.Clear();
            _denseSlotIndices.Clear();
            _sparseToDense.Clear();
            _sparseToDense.Add(-1);

            _nextFreshSideTableSlot = 1;
            _sideTableLength = 1;
            _liveCount = 0;
        }

        enum PageKind : byte
        {
            Null = 0,
            Bucket = 1,
            SingleSlot = 2, // huge or external; indistinguishable in the wire format
        }

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        struct SerializedHeapHeader
        {
            public int Version;
            public int LiveCount;
            public int SideTableLength;
            public int NextFreshSideTableSlot;
            public int NumPages;
        }

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        struct SerializedPageHeader
        {
            public byte Kind;
            public int SlotSize;
            public int SlotCount;
            public int FreeCount;
            public int Alignment;
            public int BucketIdx;
        }

        // A list used as a stack (top last) goes on the wire top first.
        static unsafe void WriteStack(
            ISerializationWriter writer,
            string name,
            NativeList<int> stack
        )
        {
            var count = stack.Length;
            writer.BlitWriteRawBytes(name + "Count", &count, sizeof(int));
            var arr = new NativeArray<int>(count, Allocator.Temp);
            for (int i = 0; i < count; i++)
                arr[i] = stack[count - 1 - i];
            writer.BlitWriteRawBytes(
                name,
                NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(arr),
                count * UnsafeUtility.SizeOf<int>()
            );
            arr.Dispose();
        }

        static unsafe void ReadStack(
            ISerializationReader reader,
            string name,
            ref NativeList<int> stack
        )
        {
            int count;
            reader.BlitReadRawBytes(name + "Count", &count, sizeof(int));
            var arr = new NativeArray<int>(count, Allocator.Temp);
            reader.BlitReadRawBytes(
                name,
                NativeArrayUnsafeUtility.GetUnsafePtr(arr),
                count * UnsafeUtility.SizeOf<int>()
            );
            // Data is top-to-bottom; push in reverse so top ends up on top.
            for (int i = count - 1; i >= 0; i--)
                stack.Add(arr[i]);
            arr.Dispose();
        }

        unsafe void WriteSideTableAndFreeSlots(ISerializationWriter writer)
        {
            var liveCountLocal = _liveCount;
            writer.BlitWriteRawBytes("NumLiveEntries", &liveCountLocal, sizeof(int));

            TrecsDebugAssert.That(
                _densePayloads.Length == _liveCount,
                "Serialize: dense-list length {0} vs _liveCount {1}",
                _densePayloads.Length,
                _liveCount
            );

            writer.BlitWriteRawBytes(
                "SlotIndices",
                _denseSlotIndices.GetUnsafeReadOnlyPtr(),
                _liveCount * sizeof(int)
            );
            writer.BlitWriteRawBytes(
                "Entries",
                _densePayloads.GetUnsafeReadOnlyPtr(),
                _liveCount * UnsafeUtility.SizeOf<NativeHeapEntryPayload>()
            );
            var typeHashes = new NativeArray<int>(
                _liveCount,
                Allocator.Temp,
                NativeArrayOptions.UninitializedMemory
            );
            var generations = new NativeArray<byte>(
                _liveCount,
                Allocator.Temp,
                NativeArrayOptions.UninitializedMemory
            );
            for (int n = 0; n < _liveCount; n++)
            {
                var entry = GetEntryPtr(_denseSlotIndices[n]);
                typeHashes[n] = entry->TypeHash;
                generations[n] = entry->Generation;
            }
            writer.BlitWriteRawBytes(
                "TypeHashes",
                typeHashes.GetUnsafeReadOnlyPtr(),
                _liveCount * sizeof(int)
            );
            writer.BlitWriteRawBytes("Generations", generations.GetUnsafeReadOnlyPtr(), _liveCount);
            typeHashes.Dispose();
            generations.Dispose();

            WriteSideTableFreeSlots(writer);
            WriteStack(writer, "freePageIds", _store.FreePageIds);

            // Batch all bucket free-slot pairs into two contiguous arrays
            // to minimize NativeArray allocations (was 2 per bucket = 26 allocs).
            int totalBucketFreeSlots = 0;
            for (int b = 0; b < BucketCount; b++)
                totalBucketFreeSlots += _store.Buckets[b].FreeSlots.Length;

            var allBucketPageIds = new NativeArray<int>(totalBucketFreeSlots, Allocator.Temp);
            var allBucketSlotIdxs = new NativeArray<int>(totalBucketFreeSlots, Allocator.Temp);
            var allBucketPageIdsPtr = (int*)NativeArrayUnsafeUtility.GetUnsafePtr(allBucketPageIds);
            var allBucketSlotIdxsPtr = (int*)
                NativeArrayUnsafeUtility.GetUnsafePtr(allBucketSlotIdxs);
            int offset = 0;

            for (int b = 0; b < BucketCount; b++)
            {
                var freeSlots = _store.Buckets[b].FreeSlots;
                var count = freeSlots.Length;
                writer.BlitWriteRawBytes("BucketFreeSlotsCount", &count, sizeof(int));
                // Top of the stack first.
                for (int i = count - 1; i >= 0; i--)
                {
                    allBucketPageIdsPtr[offset] = freeSlots[i].x;
                    allBucketSlotIdxsPtr[offset] = freeSlots[i].y;
                    offset++;
                }
                writer.BlitWriteRawBytes(
                    "PageIds",
                    allBucketPageIdsPtr + (offset - count),
                    count * sizeof(int)
                );
                writer.BlitWriteRawBytes(
                    "SlotIndices",
                    allBucketSlotIdxsPtr + (offset - count),
                    count * sizeof(int)
                );
            }
            allBucketPageIds.Dispose();
            allBucketSlotIdxs.Dispose();
        }

        unsafe void WriteSideTableFreeSlots(ISerializationWriter writer)
        {
            var count = _freeSideTableSlots.Count;
            writer.BlitWriteRawBytes("FreeSideTableSlotsCount", &count, sizeof(int));
            var slotIdxArr = new NativeArray<int>(count, Allocator.Temp);
            var genArr = new NativeArray<byte>(count, Allocator.Temp);
            int idx = 0;
            foreach (var item in _freeSideTableSlots)
            {
                slotIdxArr[idx] = item;
                genArr[idx] = GetEntry(item).Generation;
                idx++;
            }
            writer.BlitWriteRawBytes(
                "SlotIndices",
                NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(slotIdxArr),
                count * UnsafeUtility.SizeOf<int>()
            );
            writer.BlitWriteRawBytes(
                "Generations",
                NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(genArr),
                count * UnsafeUtility.SizeOf<byte>()
            );
            slotIdxArr.Dispose();
            genArr.Dispose();
        }

        unsafe void ReadSideTableFreeSlots(ISerializationReader reader)
        {
            int count;
            reader.BlitReadRawBytes("FreeSideTableSlotsCount", &count, sizeof(int));
            var slotIdxArr = new NativeArray<int>(count, Allocator.Temp);
            var genArr = new NativeArray<byte>(count, Allocator.Temp);
            reader.BlitReadRawBytes(
                "SlotIndices",
                NativeArrayUnsafeUtility.GetUnsafePtr(slotIdxArr),
                count * UnsafeUtility.SizeOf<int>()
            );
            reader.BlitReadRawBytes(
                "Generations",
                NativeArrayUnsafeUtility.GetUnsafePtr(genArr),
                count * UnsafeUtility.SizeOf<byte>()
            );
            // Data is top-to-bottom; push in reverse so top ends up on top.
            for (int i = count - 1; i >= 0; i--)
            {
                var slotIdx = slotIdxArr[i];
                TrecsDebugAssert.That(
                    _sparseToDense[slotIdx] == -1,
                    "ReadSideTableFreeSlots: slot {0} marked free but _sparseToDense is {1}",
                    slotIdx,
                    _sparseToDense[slotIdx]
                );
                _freeSideTableSlots.Push(slotIdx);
                // Overwrite the full slot, not just Generation. The slot may
                // have been *live* in the old world we just replaced (with
                // InUse=1 and a stale Address from a now-freed page); we
                // need to bring it back to a clean free state. Other fields
                // default to zero — Alloc overwrites them on slot reuse.
                // Pairs with the partial-wipe scoping in
                // ResetForDeserialize: slots below newNextFreshSlot are no
                // longer cleared by the reset, so the deserialize-side write
                // here is the only path that resets their state.
                SetEntry(slotIdx, new NativeHeapEntry { Generation = genArr[i] });
            }
            slotIdxArr.Dispose();
            genArr.Dispose();
        }

        // ─── Internal mechanics ──────────────────────────────────

        /// <summary>
        /// Registers an externally-allocated pointer as a single-slot page owned by this
        /// chunk store. The pointer must have been allocated via <c>AllocatorManager.Allocate</c>
        /// with the supplied <paramref name="size"/> and <paramref name="alignment"/> — those
        /// values are stored verbatim and used to call <c>AllocatorManager.Free</c> when the
        /// returned handle is freed. After this call the chunk store takes exclusive ownership.
        /// </summary>
        public PtrHandle AllocExternal(
            IntPtr address,
            int size,
            int alignment,
            int typeHash,
            [CallerFilePath] string callerFile = null,
            [CallerLineNumber] int callerLine = 0
        )
        {
            using var _ = TrecsProfiling.Start("NativeHeap.AllocExternal");
            TrecsDebugAssert.That(!_isDisposed);
            TrecsDebugAssert.That(address != IntPtr.Zero, "AllocExternal: null address");
            TrecsDebugAssert.That(size > 0, "AllocExternal: size must be positive");
            TrecsDebugAssert.That(
                alignment > 0 && (alignment & (alignment - 1)) == 0,
                "AllocExternal: alignment {0} must be a positive power of two",
                alignment
            );
            TrecsDebugAssert.That(
                UnityThreadHelper.IsMainThread,
                "NativeHeap.AllocExternal must be called from the main thread"
            );

#if ENABLE_UNITY_COLLECTIONS_CHECKS
            TrecsDebugAssert.That(
                _externalAddresses.Add(address),
                "AllocExternal: pointer 0x{0} is already registered in this chunk store "
                    + "(double-take-ownership leaves two aliased slots; freeing either "
                    + "dangles the other)",
                address.ToInt64()
            );
#endif

            var page = new Page
            {
                Address = address,
                BucketIdx = -1,
                SlotSize = size,
                SlotCount = 1,
                FreeCount = 0,
                Alignment = alignment,
            };
            var pageId = _store.AcquirePageId(page);

            var sideTableIdx = AcquireSideTableSlot();
            var prior = GetEntry(sideTableIdx);
            var nextGen = (byte)(prior.Generation + 1);
            if (nextGen == 0)
                nextGen = 1;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
            var safety = AtomicSafetyHandle.Create();
            AtomicSafetyHandle.SetBumpSecondaryVersionOnScheduleWrite(safety, true);
#endif

            var entry = new NativeHeapEntry
            {
                Address = address,
                TypeHash = typeHash,
                Generation = nextGen,
                InUse = 1,
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Safety = safety,
#endif
            };

            var handleValue = NativeHeapResolver.EncodeHandleValue(nextGen, (uint)sideTableIdx);
            // See the Alloc method for the rationale; the slot was free and no
            // reader has a handle for it yet.
            SetEntry(sideTableIdx, entry);
            _liveCount++;

            _sparseToDense[sideTableIdx] = _densePayloads.Length;
            _denseSlotIndices.Add(sideTableIdx);
            _densePayloads.Add(new NativeHeapEntryPayload { PageId = pageId, OwnsWholePage = 1 });

#if DEBUG
            if (callerFile != null)
            {
                _callerInfoByHandle[handleValue] = (callerFile, callerLine);
            }
#endif

            return new PtrHandle(handleValue);
        }

        void ReturnSlot(int pageId, int slotIdx, bool ownsWholePage)
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            // An external page may be registered again once released.
            if (ownsWholePage)
            {
                _externalAddresses.Remove(_store.Pages[pageId].Address);
            }
#endif
            _store.ReturnSlot(pageId, slotIdx, ownsWholePage);
        }

        int AcquireSideTableSlot()
        {
            if (_freeSideTableSlots.Count > 0)
            {
                var candidate = _freeSideTableSlots.Pop();
                // Materialise the side-table slot so Alloc can read its current generation
                // (the in-table Generation byte is the source of truth for slot reuse).
                EnsureSideTableLength(candidate + 1);
                return candidate;
            }

            var idx = _nextFreshSideTableSlot++;
            TrecsDebugAssert.That(
                (uint)idx <= NativeHeapResolver.MaxIndex,
                "NativeHeap exhausted side-table index space ({0} max)",
                NativeHeapResolver.MaxIndex
            );
            EnsureSideTableLength(idx + 1);
            return idx;
        }

        unsafe void EnsureSideTableLength(int requiredLength)
        {
            // Grow the chunked side table to cover at least `requiredLength` slots.
            // Materialise any chunks not yet allocated, zero-initialised so unused
            // slots read as default (InUse=0, Generation=0).
            while (_sideTableLength < requiredLength)
            {
                var chunkIdx = _sideTableLength >> NativeHeapResolver.ChunkSizeBits;
                if (_chunkDirectory[chunkIdx] == IntPtr.Zero)
                {
                    AllocateChunk(chunkIdx);
                }
                _sideTableLength++;
            }

            if (_sparseToDense.Length < requiredLength)
            {
                var oldLen = _sparseToDense.Length;
                _sparseToDense.Resize(requiredLength, NativeArrayOptions.UninitializedMemory);
                var ptr = (int*)_sparseToDense.GetUnsafePtr() + oldLen;
                UnsafeUtility.MemSet(ptr, 0xFF, (long)(requiredLength - oldLen) * sizeof(int));
            }
        }

        unsafe void AllocateChunk(int chunkIdx)
        {
            TrecsDebugAssert.That(
                chunkIdx < NativeHeapResolver.MaxChunkCount,
                "AllocateChunk: chunkIdx {0} exceeds max chunk count {1}",
                chunkIdx,
                NativeHeapResolver.MaxChunkCount
            );
            TrecsDebugAssert.That(
                _chunkDirectory[chunkIdx] == IntPtr.Zero,
                "AllocateChunk: chunk {0} already allocated",
                chunkIdx
            );

            var chunkBytes = NativeHeapResolver.ChunkByteSize;
            var ptr = AllocatorManager.Allocate(
                Allocator.Persistent,
                chunkBytes,
                UnsafeUtility.AlignOf<NativeHeapEntry>(),
                items: 1
            );
            TrecsDebugAssert.That(
                ptr != null,
                "AllocateChunk: AllocatorManager returned null for chunk {0}",
                chunkIdx
            );
            UnsafeUtility.MemClear(ptr, chunkBytes);
            // Defense-in-depth: any handle obtained via a normal Alloc → Schedule path
            // already sees the zero-init because Unity's job-system Schedule() fences
            // main-thread writes before the worker runs. The barrier here covers the
            // resolver's contract that ANY PtrHandle (garbage, stale, synthesised) must
            // safely report "not live" — without it, a handle whose index lands in a
            // mid-materialised chunk could, on weakly-ordered hardware (ARM), see the
            // new directory pointer but read uninitialised slot bytes that happen to
            // pass InUse=1/Generation checks. Also serves as a compiler fence on x64.
            Thread.MemoryBarrier();
            _chunkDirectory[chunkIdx] = new IntPtr(ptr);
        }

        unsafe NativeHeapEntry* GetEntryPtr(int idx)
        {
            TrecsDebugAssert.That(
                idx >= 0 && idx < _sideTableLength,
                "GetEntryPtr: idx {0} out of range (length {1})",
                idx,
                _sideTableLength
            );
            var chunkPtr = (NativeHeapEntry*)
                _chunkDirectory[idx >> NativeHeapResolver.ChunkSizeBits].ToPointer();
            return chunkPtr + (idx & NativeHeapResolver.ChunkIndexMask);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe NativeHeapEntry GetEntry(int idx)
        {
            TrecsDebugAssert.That(
                idx >= 0 && idx < _sideTableLength,
                "GetEntry: idx {0} out of range (length {1})",
                idx,
                _sideTableLength
            );
            var chunkIdx = idx >> NativeHeapResolver.ChunkSizeBits;
            var chunkPtr = (NativeHeapEntry*)_chunkDirectory[chunkIdx].ToPointer();
            return chunkPtr[idx & NativeHeapResolver.ChunkIndexMask];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe void SetEntry(int idx, in NativeHeapEntry entry)
        {
            TrecsDebugAssert.That(
                idx >= 0 && idx < _sideTableLength,
                "SetEntry: idx {0} out of range (length {1})",
                idx,
                _sideTableLength
            );
            var chunkIdx = idx >> NativeHeapResolver.ChunkSizeBits;
            var chunkPtr = (NativeHeapEntry*)_chunkDirectory[chunkIdx].ToPointer();
            chunkPtr[idx & NativeHeapResolver.ChunkIndexMask] = entry;
        }

        unsafe void DisposeAllChunks()
        {
            for (int i = 0; i < _chunkDirectory.Length; i++)
            {
                var ptr = _chunkDirectory[i];
                if (ptr == IntPtr.Zero)
                {
                    continue;
                }
                AllocatorManager.Free(
                    Allocator.Persistent,
                    ptr.ToPointer(),
                    NativeHeapResolver.ChunkByteSize,
                    UnsafeUtility.AlignOf<NativeHeapEntry>(),
                    items: 1
                );
                _chunkDirectory[i] = IntPtr.Zero;
            }
        }

        // ─── Inner types ─────────────────────────────────────────

        struct Bucket
        {
            public int SlotSize;
            public int SlotsPerPage;
            public int PageSizeBytes;
            public UnsafeList<int> PageIds;

            // Stack: top is last.
            public UnsafeList<int2> FreeSlots;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Page
        {
            public IntPtr Address;
            public int BucketIdx; // -1 for huge or external single-slot pages
            public int SlotSize;
            public int SlotCount;
            public int FreeCount;
            public int Alignment;
        }

        /// <summary>
        /// Pages and their free slots. Plain shared containers, so a copy works on the same
        /// state: the managed paths and the Burst adoption job run this one code.
        /// </summary>
        unsafe struct PageStore
        {
            public NativeList<Page> Pages;

            // Stack: top is last.
            public NativeList<int> FreePageIds;

            [NativeDisableUnsafePtrRestriction]
            public Bucket* Buckets;

            public static PageStore Create()
            {
                var store = new PageStore
                {
                    Pages = new NativeList<Page>(64, Allocator.Persistent),
                    FreePageIds = new NativeList<int>(16, Allocator.Persistent),
                    Buckets = (Bucket*)
                        UnsafeUtility.MallocTracked(
                            sizeof(Bucket) * BucketCount,
                            UnsafeUtility.AlignOf<Bucket>(),
                            Allocator.Persistent,
                            0
                        ),
                };
                for (int i = 0; i < BucketCount; i++)
                {
                    var slotSize = MinBucketSlotSize << i;
                    var pageSize = Math.Max(MinPageSize, slotSize * TargetSlotsPerPage);
                    store.Buckets[i] = new Bucket
                    {
                        SlotSize = slotSize,
                        SlotsPerPage = pageSize / slotSize,
                        PageSizeBytes = pageSize,
                        PageIds = new UnsafeList<int>(8, Allocator.Persistent),
                        FreeSlots = new UnsafeList<int2>(64, Allocator.Persistent),
                    };
                }
                return store;
            }

            /// <summary>Frees every page and forgets every slot.</summary>
            public void Clear()
            {
                for (int i = 0; i < Pages.Length; i++)
                {
                    var page = Pages[i];
                    if (page.Address != IntPtr.Zero)
                    {
                        FreePage(in page);
                    }
                }
                Pages.Clear();
                FreePageIds.Clear();
                for (int b = 0; b < BucketCount; b++)
                {
                    Buckets[b].PageIds.Clear();
                    Buckets[b].FreeSlots.Clear();
                }
            }

            public void Dispose()
            {
                Clear();
                for (int b = 0; b < BucketCount; b++)
                {
                    Buckets[b].PageIds.Dispose();
                    Buckets[b].FreeSlots.Dispose();
                }
                UnsafeUtility.FreeTracked(Buckets, Allocator.Persistent);
                Pages.Dispose();
                FreePageIds.Dispose();
            }

            public void AllocSlot(
                int size,
                int alignment,
                out int pageId,
                out int slotIdx,
                out int bucketIdx,
                out bool ownsWholePage,
                out int slotByteSize,
                out IntPtr address
            )
            {
                var neededSlotSize = Math.Max(size, alignment);
                if (neededSlotSize > MaxBucketSlotSize)
                {
                    var pagePtr = AllocatorManager.Allocate(
                        Allocator.Persistent,
                        neededSlotSize,
                        alignment,
                        items: 1
                    );
                    TrecsDebugAssert.That(
                        pagePtr != null,
                        "AllocatorManager returned null huge page (size {0})",
                        neededSlotSize
                    );
                    address = new IntPtr(pagePtr);
                    pageId = AcquirePageId(
                        new Page
                        {
                            Address = address,
                            BucketIdx = -1,
                            SlotSize = neededSlotSize,
                            SlotCount = 1,
                            FreeCount = 0,
                            Alignment = alignment,
                        }
                    );
                    slotIdx = 0;
                    bucketIdx = -1;
                    ownsWholePage = true;
                    slotByteSize = neededSlotSize;
                    return;
                }

                bucketIdx = math.max(
                    0,
                    math.ceillog2(neededSlotSize) - math.tzcnt(MinBucketSlotSize)
                );
                ref var bucket = ref Buckets[bucketIdx];
                if (bucket.FreeSlots.Length == 0)
                {
                    AddPage(bucketIdx);
                }
                var top = bucket.FreeSlots.Length - 1;
                var free = bucket.FreeSlots[top];
                bucket.FreeSlots.Length = top;

                var page = Pages[free.x];
                TrecsDebugAssert.That(
                    page.Address != IntPtr.Zero && page.FreeCount > 0,
                    "Free slot pointed at reclaimed or full page {0}",
                    free.x
                );
                page.FreeCount--;
                Pages[free.x] = page;

                pageId = free.x;
                slotIdx = free.y;
                ownsWholePage = false;
                slotByteSize = bucket.SlotSize;
                address = new IntPtr((byte*)page.Address.ToPointer() + free.y * page.SlotSize);
            }

            void AddPage(int bucketIdx)
            {
                ref var bucket = ref Buckets[bucketIdx];
                var pagePtr = AllocatorManager.Allocate(
                    Allocator.Persistent,
                    bucket.PageSizeBytes,
                    bucket.SlotSize,
                    items: 1
                );
                TrecsDebugAssert.That(
                    pagePtr != null,
                    "AllocatorManager returned null page (size {0}, align {1})",
                    bucket.PageSizeBytes,
                    bucket.SlotSize
                );
                var pageId = AcquirePageId(
                    new Page
                    {
                        Address = new IntPtr(pagePtr),
                        BucketIdx = bucketIdx,
                        SlotSize = bucket.SlotSize,
                        SlotCount = bucket.SlotsPerPage,
                        FreeCount = bucket.SlotsPerPage,
                        Alignment = bucket.SlotSize,
                    }
                );
                bucket.PageIds.Add(pageId);
                // Reverse, so pops come out in ascending slot order.
                for (int s = bucket.SlotsPerPage - 1; s >= 0; s--)
                {
                    bucket.FreeSlots.Add(new int2(pageId, s));
                }
            }

            public int AcquirePageId(Page page)
            {
                var freeCount = FreePageIds.Length;
                if (freeCount > 0)
                {
                    var pageId = FreePageIds[freeCount - 1];
                    FreePageIds.Length = freeCount - 1;
                    Pages[pageId] = page;
                    return pageId;
                }
                Pages.Add(page);
                return Pages.Length - 1;
            }

            public void ReturnSlot(int pageId, int slotIdx, bool ownsWholePage)
            {
                var page = Pages[pageId];
                TrecsDebugAssert.That(
                    page.Address != IntPtr.Zero,
                    "ReturnSlot: page {0} already reclaimed",
                    pageId
                );

                if (ownsWholePage)
                {
                    FreePage(in page);
                    Pages[pageId] = default;
                    FreePageIds.Add(pageId);
                    return;
                }

                page.FreeCount++;
                TrecsDebugAssert.That(
                    page.FreeCount <= page.SlotCount,
                    "Page {0} free-count overflow ({1} > {2})",
                    pageId,
                    page.FreeCount,
                    page.SlotCount
                );
                Pages[pageId] = page;
                Buckets[page.BucketIdx].FreeSlots.Add(new int2(pageId, slotIdx));
            }

            /// <summary>Frees an empty page; its free slots leave the bucket in order.</summary>
            public void ReclaimPage(int pageId)
            {
                var page = Pages[pageId];
                if (page.BucketIdx >= 0)
                {
                    ref var bucket = ref Buckets[page.BucketIdx];
                    var pageIdIndex = bucket.PageIds.IndexOf(pageId);
                    bucket.PageIds.RemoveAt(pageIdIndex);

                    var kept = 0;
                    for (int i = 0; i < bucket.FreeSlots.Length; i++)
                    {
                        var free = bucket.FreeSlots[i];
                        if (free.x != pageId)
                        {
                            bucket.FreeSlots[kept++] = free;
                        }
                    }
                    bucket.FreeSlots.Length = kept;
                }
                FreePage(in page);
                Pages[pageId] = default;
                FreePageIds.Add(pageId);
            }

            static void FreePage(in Page page)
            {
                var sizeOf = page.BucketIdx >= 0 ? page.SlotSize * page.SlotCount : page.SlotSize;
                AllocatorManager.Free(
                    Allocator.Persistent,
                    page.Address.ToPointer(),
                    sizeOf,
                    page.Alignment,
                    items: 1
                );
            }
        }

        [BurstCompile]
        unsafe struct DeserializeSideTableJob : IJobParallelFor
        {
            [ReadOnly]
            public NativeList<int> DenseSlotIndices;

            [ReadOnly]
            public NativeList<NativeHeapEntryPayload> DensePayloads;

            [ReadOnly]
            public NativeArray<int> TypeHashes;

            [ReadOnly]
            public NativeArray<byte> Generations;

            [ReadOnly]
            public NativeList<Page> Pages;

            [NativeDisableUnsafePtrRestriction]
            [ReadOnly]
            public NativeArray<IntPtr> ChunkDirectory;

            [NativeDisableContainerSafetyRestriction]
            [NativeDisableParallelForRestriction]
            public NativeList<int> SparseToDense;

            public void Execute(int n)
            {
                var slotIdx = DenseSlotIndices[n];
                var payload = DensePayloads[n];

                var page = Pages[payload.PageId];
                var address = new IntPtr(
                    (byte*)page.Address.ToPointer() + payload.SlotIndex * page.SlotSize
                );

                var entry = new NativeHeapEntry
                {
                    Address = address,
                    TypeHash = TypeHashes[n],
                    Generation = Generations[n],
                    InUse = 1,
                };

                var chunkIdx = slotIdx >> NativeHeapResolver.ChunkSizeBits;
                var chunkPtr = (NativeHeapEntry*)ChunkDirectory[chunkIdx].ToPointer();
                chunkPtr[slotIdx & NativeHeapResolver.ChunkIndexMask] = entry;

                SparseToDense[slotIdx] = n;
            }
        }
    }

    /// <summary>
    /// Snapshot of <see cref="NativeHeap"/> state at one point in time. Returned by
    /// <c>NativeHeap.GetStatistics</c>. Intended for diagnostics / benchmarks only;
    /// not part of the public Trecs API surface.
    /// </summary>
    internal struct HeapStatistics
    {
        public int LiveAllocations;
        public int NumPages;
        public int NumHugePages;
        public int SideTableLength;
        public int FreeSideTableSlots;
        public int NextFreshSideTableSlot;
        public long HugePagesReservedBytes;
        public long TotalReservedBytes;
        public BucketStatistics[] Buckets;
    }

    internal struct BucketStatistics
    {
        public int SlotSize;
        public int SlotsPerPage;
        public int PageCount;
        public int LiveSlotCount;
        public int FreeSlotCount;
        public long ReservedBytes;
    }

    /// <summary>
    /// One row of <see cref="NativeHeap.SnapshotPageOccupancy"/> output.
    /// <c>LiveSlotsStartIndex</c> indexes into the parallel <c>List&lt;bool&gt;</c>
    /// the caller passed in; <c>SlotCount</c> bools follow, one per slot, true if
    /// the slot is currently in use.
    /// </summary>
    internal struct PageOccupancyHeader
    {
        public int PageId;
        public int BucketIdx; // -1 for huge / external single-slot pages
        public int SlotSize;
        public int SlotCount;
        public int FreeCount;
        public int LiveSlotsStartIndex;
    }
}
