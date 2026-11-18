using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Trecs.Internal
{
    /// <summary>
    /// Side-table entry for a single chunk-store allocation. One entry per <see cref="PtrHandle"/>
    /// the store has handed out; recycled when the handle is freed. The <see cref="Generation"/>
    /// field plus the handle's encoded generation detect stale-handle access after slot reuse.
    ///
    /// <para>Only fields needed by Burst jobs live here. Page bookkeeping (OwnsWholePage, PageId,
    /// SlotIndex) lives in the dense <see cref="NativeHeapEntryPayload"/> list, keeping the
    /// entry at 16 bytes without safety checks. Each side-table chunk ends with one detach bit
    /// per entry, outside the entries.</para>
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public struct NativeHeapEntry
    {
        /// <summary>Current physical address of the slot. Stable while this entry is live.</summary>
        public IntPtr Address;

        /// <summary>Burst-compatible type hash for the value stored at this slot. 0 means untyped.</summary>
        public int TypeHash;

        /// <summary>
        /// Increments every time this side-table slot transitions (alloc, free).
        /// A handle is valid iff its encoded generation equals this value AND <see cref="InUse"/> is 1.
        /// Wraps around through the byte range, skipping 0 (which means "never allocated").
        /// </summary>
        public byte Generation;

        /// <summary>1 if this slot currently backs a live allocation; 0 otherwise.</summary>
        public byte InUse;

        /// <summary>
        /// 1 while <see cref="Address"/> points at a block a job grew outside the heap's pages
        /// (<see cref="DetachedBlock"/>). The heap's own slot for the allocation stays reserved
        /// until <c>NativeHeap.AdoptDetached</c> moves the block back into it.
        /// </summary>
        public byte IsDetached;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        /// <summary>
        /// Per-allocation safety handle. Stable across the lifetime of this allocation:
        /// minted on alloc, released on free, never re-keyed by compaction (compaction
        /// only updates <see cref="Address"/>).
        /// </summary>
        public AtomicSafetyHandle Safety;
#endif
    }

    /// <summary>
    /// Where a live allocation sits in the pages: main-thread bookkeeping, kept out of
    /// <see cref="NativeHeapEntry"/> so jobs resolve through 16-byte entries. Type hash and
    /// generation live in the entry only. Serialized as is.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeHeapEntryPayload
    {
        public int PageId;

        // A bucket page holds at most MinPageSize / MinBucketSlotSize slots.
        public ushort SlotIndex;
        public byte OwnsWholePage;
        public byte _padding;
    }

    /// <summary>
    /// Burst-compatible resolver over the chunk store's side table. Cheap to copy by value into
    /// job structs. Resolution is two indexed loads (chunk-directory entry → entry within
    /// chunk) plus a generation/in-use check: no hash lookup, no dictionary probe.
    ///
    /// <para>The side table is stored as a fixed-capacity directory of pointers to
    /// fixed-size chunks of <see cref="NativeHeapEntry"/>. New chunks are appended
    /// when slot indices cross a chunk boundary; existing chunks never move. That stability
    /// is what lets main-thread <c>Alloc</c> materialise new slots concurrent with jobs
    /// reading via this resolver — the list-growth-moves-the-backing-buffer race that a
    /// flat <c>NativeList</c> would have is eliminated by design.</para>
    /// </summary>
    [EditorBrowsable(EditorBrowsableState.Never)]
    public readonly struct NativeHeapResolver
    {
        // Bit layout of the 32-bit PtrHandle.Value encoding the chunk-store identity.
        //   bits  0..23 → side-table index (1..0xFFFFFF; 0 reserved for null)
        //   bits 24..31 → generation       (1..255; 0 reserved for "never allocated")
        internal const int IndexBits = 24;
        internal const uint IndexMask = (1u << IndexBits) - 1u;
        internal const int GenerationShift = IndexBits;
        internal const uint MaxIndex = IndexMask; // 0xFFFFFF == 16,777,215

        // Chunked side-table layout. Each chunk is a fixed-size block of
        // NativeHeapEntry; the directory is a fixed-capacity array of pointers
        // to those chunks (or IntPtr.Zero for chunks not yet materialised).
        internal const int ChunkSizeBits = 10;
        internal const int ChunkSize = 1 << ChunkSizeBits; // 1024 entries per chunk
        internal const int ChunkIndexMask = ChunkSize - 1;

        // Total chunk slots needed to address every possible side-table index.
        internal const int MaxChunkCount = (int)((MaxIndex + 1u) >> ChunkSizeBits); // 16,384

        // Bytes of one side-table chunk: its entries, then one detached bit per entry.
        internal static int ChunkByteSize =>
            UnsafeUtility.SizeOf<NativeHeapEntry>() * ChunkSize + ChunkSize / 8;

        [NativeDisableContainerSafetyRestriction]
        readonly NativeArray<IntPtr> _chunkDirectory;

        [NativeDisableUnsafePtrRestriction]
        readonly unsafe NativeHeapDetachLog* _detachLog;

        // 1 if this resolver may be passed to a heap-mutating Write open
        // (TrecsList, NativeUniquePtr); 0 if Read-only. Stamped from the role
        // when WorldAccessor / NativeWorldAccessor hand the resolver out. The
        // raw resolver owned by NativeHeap itself is permissive (1)
        // because it's only consumed by already-gated internal paths.
        readonly byte _canMutateHeap;

        // Internal: only NativeHeap constructs the raw resolver. Role-aware
        // copies for the user-facing surface go through the (in source, bool)
        // overload below.
        internal unsafe NativeHeapResolver(
            NativeArray<IntPtr> chunkDirectory,
            NativeHeapDetachLog* detachLog
        )
        {
            _chunkDirectory = chunkDirectory;
            _detachLog = detachLog;
            _canMutateHeap = 1;
        }

        // Role-aware copy: same backing directory, override the mutation bit.
        // Used by WorldAccessor and NativeWorldAccessor to produce a Variable-role
        // resolver that fails fast at Write-open time inside Burst jobs.
        internal unsafe NativeHeapResolver(in NativeHeapResolver source, bool canMutateHeap)
        {
            _chunkDirectory = source._chunkDirectory;
            _detachLog = source._detachLog;
            _canMutateHeap = canMutateHeap ? (byte)1 : (byte)0;
        }

        internal unsafe NativeHeapDetachLog* DetachLog => _detachLog;

        /// <summary>
        /// Asserts the allocation <paramref name="handle"/> named is still live in
        /// <paramref name="slot"/>: catches a view used after its container was disposed,
        /// in shipping builds too (1-byte generation: 1/256 chance to miss once reused).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void AssertSlotLive(NativeHeapEntry* slot, PtrHandle handle)
        {
            TrecsAssert.That(
                slot->InUse == 1 && slot->Generation == (byte)(handle.Value >> GenerationShift),
                "Heap view is stale: allocation {0} was freed since the view was opened. "
                    + "Re-open the view after any code path that disposes the container.",
                handle.Value
            );
        }

        /// <summary>
        /// Slot size the heap hands out for an allocation of <paramref name="size"/> bytes:
        /// a power-of-two bucket up to <see cref="NativeHeap.MaxBucketSlotSize"/>, the exact
        /// size above it. Growable containers size their capacity to fill it.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int GetSlotByteSize(int size, int alignment)
        {
            var needed = math.max(size, alignment);
            return needed > NativeHeap.MaxBucketSlotSize
                ? needed
                : math.max(NativeHeap.MinBucketSlotSize, math.ceilpow2(needed));
        }

        /// <summary>
        /// Burst-job-side gate for heap mutation through this resolver. Called from
        /// <see cref="NativeUniquePtr{T}"/>'s <c>Write(in NativeHeapResolver)</c>
        /// and from <see cref="TrecsList{T}"/>'s <c>Write(in NativeHeapResolver)</c>.
        /// The flag is stamped from the originating accessor's role.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AssertCanMutateHeap()
        {
            TrecsAssert.That(
                _canMutateHeap == 1,
                "Attempted heap mutation (Write) on a heap pointer through a "
                    + "Variable-role NativeHeapResolver. Heap mutation is "
                    + "only allowed from Fixed-role and Unrestricted-role accessors. "
                    + "Read access is always allowed."
            );
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public NativeHeapEntry ResolveEntry(PtrHandle handle)
        {
            TrecsAssert.That(
                TryResolveEntry(handle, out var entry),
                "Could not resolve PtrHandle {0} through chunk store "
                    + "(null, out of range, freed, or stale generation)",
                handle.Value
            );
            return entry;
        }

        /// <summary>
        /// Resolves a handle without throwing on miss. Returns true and populates <paramref name="entry"/>
        /// if the handle is live and the generation matches; returns false otherwise. Use this when the
        /// caller wants to fall back to an alternative lookup (e.g. chained chunk stores) instead of
        /// failing the resolve outright.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe bool TryResolveEntry(PtrHandle handle, out NativeHeapEntry entry)
        {
            if (handle.IsNull)
            {
                entry = default;
                return false;
            }

            DecodeHandle(handle, out var index, out var generation);

            var chunkIdx = (int)index >> ChunkSizeBits;
            if (chunkIdx >= _chunkDirectory.Length)
            {
                entry = default;
                return false;
            }

            var chunkPtr = (NativeHeapEntry*)_chunkDirectory[chunkIdx].ToPointer();
            if (chunkPtr == null)
            {
                // Chunk not yet materialised — the handle's index is past the
                // current high-water mark from this resolver's point of view.
                entry = default;
                return false;
            }

            entry = chunkPtr[(int)index & ChunkIndexMask];
            return entry.InUse == 1 && entry.Generation == generation;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe void* ResolveUnsafePtr(PtrHandle handle)
        {
            return ResolveEntry(handle).Address.ToPointer();
        }

        /// <summary>
        /// Resolves a handle and returns the address of the backing
        /// <see cref="NativeHeapEntry"/> slot in the side table, in addition
        /// to a copy of the entry's current state. Used by Read/Write wrappers
        /// that need a stable pointer to the slot so they can re-check the
        /// slot's <see cref="NativeHeapEntry.Generation"/> on every
        /// access — the shipping-build use-after-dispose guard. Chunks never
        /// move once allocated, so the returned pointer is stable for the
        /// chunk store's lifetime.
        ///
        /// <para>Throws if the handle is null, the chunk hasn't been
        /// materialised yet, the slot's InUse bit is 0, or the slot's
        /// generation doesn't match the handle's encoded generation — same
        /// validity rules as <see cref="ResolveEntry"/>.</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe NativeHeapEntry ResolveEntryWithSlotPtr(
            PtrHandle handle,
            out NativeHeapEntry* slotPtr
        )
        {
            TrecsAssert.That(
                TryResolveEntryWithSlotPtr(handle, out var entry, out slotPtr),
                "Could not resolve PtrHandle {0} through chunk store "
                    + "(null, out of range, freed, or stale generation)",
                handle.Value
            );
            return entry;
        }

        /// <summary>
        /// Non-throwing counterpart to <see cref="ResolveEntryWithSlotPtr"/>.
        /// On success returns true and populates both <paramref name="entry"/>
        /// (the entry's value at resolve time) and <paramref name="slotPtr"/>
        /// (the address of that entry in the side-table chunk; stable for the
        /// chunk store's lifetime). On failure both outs are zeroed.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe bool TryResolveEntryWithSlotPtr(
            PtrHandle handle,
            out NativeHeapEntry entry,
            out NativeHeapEntry* slotPtr
        )
        {
            if (handle.IsNull)
            {
                entry = default;
                slotPtr = null;
                return false;
            }

            DecodeHandle(handle, out var index, out var generation);

            var chunkIdx = (int)index >> ChunkSizeBits;
            if (chunkIdx >= _chunkDirectory.Length)
            {
                entry = default;
                slotPtr = null;
                return false;
            }

            var chunkPtr = (NativeHeapEntry*)_chunkDirectory[chunkIdx].ToPointer();
            if (chunkPtr == null)
            {
                entry = default;
                slotPtr = null;
                return false;
            }

            slotPtr = chunkPtr + ((int)index & ChunkIndexMask);
            entry = *slotPtr;
            return entry.InUse == 1 && entry.Generation == generation;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint EncodeHandleValue(uint generation, uint index)
        {
            TrecsDebugAssert.That(
                index >= 1 && index <= MaxIndex,
                "Side-table index {0} out of range",
                index
            );
            TrecsDebugAssert.That(
                generation >= 1 && generation <= 0xFF,
                "Generation {0} out of range",
                generation
            );
            return (generation << GenerationShift) | (index & IndexMask);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void DecodeHandle(PtrHandle handle, out uint index, out byte generation)
        {
            var v = handle.Value;
            index = v & IndexMask;
            generation = (byte)(v >> GenerationShift);
        }
    }

    /// <summary>
    /// Which side-table chunks hold a detached entry: one bit per chunk. Each chunk's own
    /// bits (after its entries) name the entries. Jobs set bits; the main thread clears them
    /// when no job runs.
    /// </summary>
    internal unsafe struct NativeHeapDetachLog
    {
        internal const int ChunkWordCount = NativeHeapResolver.MaxChunkCount / 64;

        internal int HasDetached;
        internal int ArenaCount;

        // One per job thread index; detached blocks live here until the next adoption.
        internal DetachArena* Arenas;
        internal fixed long ChunkBits[ChunkWordCount];
    }

    // A cache line per thread, so threads bumping their own arena share none.
    [StructLayout(LayoutKind.Explicit, Size = 64)]
    internal unsafe struct DetachArena
    {
        // Chunks chain through their first pointer.
        [FieldOffset(0)]
        public byte* Chunk;

        [FieldOffset(8)]
        public byte* LargeChunks;

        [FieldOffset(16)]
        public int Offset;

        [FieldOffset(20)]
        public int End;
    }

    /// <summary>
    /// A heap allocation's block that a job grew, held outside the heap's pages until the
    /// next phase boundary. Placement in pages must not depend on thread timing (snapshots
    /// blit pages and lockstep peers must match), so jobs never touch pages: they bump a
    /// block from their thread's arena, repoint the allocation's side-table entry at it, and
    /// the main thread copies it into a page slot in side-table order, then frees the
    /// arenas. The handle never changes.
    /// </summary>
    internal static unsafe class DetachedBlock
    {
        // Block size sits in front of the block.
        const int PrefixBytes = 16;
        const int ChunkHeaderBytes = 16;
        const int ChunkBytes = 16 * 1024;

        // Blocks above this get a chunk of their own, leaving the shared chunk's room.
        const int LargeBlockBytes = ChunkBytes / 4;

        internal const int MaxAlignment = PrefixBytes;

        internal static int GetSize(void* block) => *(int*)((byte*)block - PrefixBytes);

        internal static void InitArenas(NativeHeapDetachLog* log)
        {
            log->ArenaCount = JobsUtility.ThreadIndexCount;
            var bytes = (long)sizeof(DetachArena) * log->ArenaCount;
            log->Arenas = (DetachArena*)
                UnsafeUtility.MallocTracked(bytes, 64, Allocator.Persistent, 0);
            UnsafeUtility.MemClear(log->Arenas, bytes);
        }

        // Main thread, no job running.
        internal static void FreeArenaChunks(NativeHeapDetachLog* log)
        {
            for (int i = 0; i < log->ArenaCount; i++)
            {
                ref var arena = ref log->Arenas[i];
                FreeChain(arena.Chunk);
                FreeChain(arena.LargeChunks);
                arena = default;
            }
        }

        internal static void DisposeArenas(NativeHeapDetachLog* log)
        {
            FreeArenaChunks(log);
            UnsafeUtility.FreeTracked(log->Arenas, Allocator.Persistent);
        }

        static void FreeChain(byte* chunk)
        {
            while (chunk != null)
            {
                var next = *(byte**)chunk;
                UnsafeUtility.Free(chunk, Allocator.Persistent);
                chunk = next;
            }
        }

        static byte* Allocate(NativeHeapDetachLog* log, int size)
        {
            var needed = PrefixBytes + ((size + 15) & ~15);
            var threadIndex = JobsUtility.ThreadIndex;
            TrecsDebugAssert.That(
                threadIndex < log->ArenaCount,
                "DetachedBlock: thread index {0} past arena count {1}",
                threadIndex,
                log->ArenaCount
            );
            ref var arena = ref log->Arenas[threadIndex];
            byte* basePtr;
            if (needed > LargeBlockBytes)
            {
                var chunk = NewChunk(ChunkHeaderBytes + needed, ref arena.LargeChunks);
                basePtr = chunk + ChunkHeaderBytes;
            }
            else
            {
                if (arena.Chunk == null || arena.Offset + needed > arena.End)
                {
                    NewChunk(ChunkBytes, ref arena.Chunk);
                    arena.Offset = ChunkHeaderBytes;
                    arena.End = ChunkBytes;
                }
                basePtr = arena.Chunk + arena.Offset;
                arena.Offset += needed;
            }
            *(int*)basePtr = size;
            return basePtr + PrefixBytes;
        }

        static byte* NewChunk(long bytes, ref byte* chain)
        {
            var chunk = (byte*)UnsafeUtility.Malloc(bytes, MaxAlignment, Allocator.Persistent);
            *(byte**)chunk = chain;
            chain = chunk;
            return chunk;
        }

        /// <summary>
        /// Moves the allocation behind <paramref name="handle"/> to a new detached block of
        /// <paramref name="newSize"/> bytes, copying its first <paramref name="copyBytes"/>
        /// bytes and zeroing the rest. Any thread; the caller holds write access to the
        /// allocation. Returns the new block.
        /// </summary>
        internal static void* Reallocate(
            NativeHeapEntry* slot,
            PtrHandle handle,
            NativeHeapDetachLog* log,
            int newSize,
            int copyBytes
        )
        {
            TrecsDebugAssert.That(
                copyBytes >= 0 && copyBytes <= newSize,
                "DetachedBlock.Reallocate: copyBytes {0} outside [0, {1}]",
                copyBytes,
                newSize
            );
            var block = Allocate(log, newSize);
            UnsafeUtility.MemCpy(block, slot->Address.ToPointer(), copyBytes);
            UnsafeUtility.MemClear(block + copyBytes, newSize - copyBytes);

            // A block grown again stays in its arena until adoption frees the arenas.
            if (slot->IsDetached == 0)
            {
                slot->IsDetached = 1;
                Mark(slot, handle, log);
            }
            slot->Address = new IntPtr(block);
            return block;
        }

        static void Mark(NativeHeapEntry* slot, PtrHandle handle, NativeHeapDetachLog* log)
        {
            NativeHeapResolver.DecodeHandle(handle, out var index, out _);
            var inChunk = (int)(index & NativeHeapResolver.ChunkIndexMask);
            var entryBits = (long*)(slot - inChunk + NativeHeapResolver.ChunkSize);
            SetBit(entryBits + (inChunk >> 6), 1L << (inChunk & 63));
            var chunk = (int)(index >> NativeHeapResolver.ChunkSizeBits);
            SetBit(log->ChunkBits + (chunk >> 6), 1L << (chunk & 63));
            // Read first: every thread writing the line would bounce it between cores.
            if (log->HasDetached == 0)
            {
                log->HasDetached = 1;
            }
        }

        static void SetBit(long* word, long bit)
        {
            var seen = *word;
            while ((seen & bit) == 0)
            {
                var prior = Interlocked.CompareExchange(ref *word, seen | bit, seen);
                if (prior == seen)
                {
                    return;
                }
                seen = prior;
            }
        }
    }
}
