using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Trecs.Internal;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Trecs.Internal
{
    /// <summary>
    /// Front of a <see cref="TrecsList{T}"/> block. Elements follow at
    /// <see cref="TrecsListLayout{T}.DataOffset"/>; the block moves when it grows.
    /// </summary>
    [EditorBrowsable(EditorBrowsableState.Never)]
    public struct TrecsListHeader
    {
        public int Count;
        public int Capacity;
    }

    /// <summary>
    /// Block layout of a <see cref="TrecsList{T}"/>: header, then elements. Capacity fills
    /// the heap slot, so a block's size is always a slot size.
    /// </summary>
    [EditorBrowsable(EditorBrowsableState.Never)]
    public static unsafe class TrecsListLayout<T>
        where T : unmanaged
    {
        public static int DataOffset
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                var align = UnsafeUtility.AlignOf<T>();
                return (sizeof(TrecsListHeader) + align - 1) & -align;
            }
        }

        public static int Alignment
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => math.max(UnsafeUtility.AlignOf<TrecsListHeader>(), UnsafeUtility.AlignOf<T>());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T* Data(TrecsListHeader* header) => (T*)((byte*)header + DataOffset);

        internal static int BlockSizeFor(int minCapacity)
        {
            var bytes = DataOffset + (long)minCapacity * sizeof(T);
            TrecsAssert.That(
                bytes <= int.MaxValue,
                "TrecsList of {0} elements of {1} bytes exceeds int.MaxValue bytes",
                minCapacity,
                sizeof(T)
            );
            return NativeHeapResolver.GetSlotByteSize((int)bytes, Alignment);
        }

        internal static int CapacityOf(int blockSize) => (blockSize - DataOffset) / sizeof(T);

        // Doubles the capacity, or takes minCapacity when doubling would pass int.MaxValue bytes.
        static int GrowBlockSize(int capacity, int minCapacity)
        {
            var doubled = math.max((long)minCapacity, 2L * capacity);
            var fits = DataOffset + doubled * sizeof(T) <= int.MaxValue;
            return BlockSizeFor(fits ? (int)doubled : minCapacity);
        }

        static int LiveBytes(TrecsListHeader* header) => DataOffset + header->Count * sizeof(T);

        internal static void InsertNoResize(TrecsListHeader* header, int index, in T value)
        {
            var count = header->Count;
            TrecsDebugAssert.That(
                (uint)index <= (uint)count,
                "TrecsList.Insert index {0} out of range (Count={1})",
                index,
                count
            );
            TrecsDebugAssert.That(
                count < header->Capacity,
                "TrecsList.InsertNoResize: list is full (capacity {0})",
                header->Capacity
            );
            // `value` may reference an element the move shifts.
            var copy = value;
            var data = Data(header);
            UnsafeUtility.MemMove(
                data + index + 1,
                data + index,
                (long)(count - index) * sizeof(T)
            );
            data[index] = copy;
            header->Count = count + 1;
        }

        internal static TrecsListHeader* Grow(
            NativeHeap heap,
            PtrHandle handle,
            TrecsListHeader* header,
            int minCapacity
        )
        {
            var size = GrowBlockSize(header->Capacity, minCapacity);
            var grown = (TrecsListHeader*)
                heap.Reallocate(handle, size, Alignment, LiveBytes(header));
            grown->Capacity = CapacityOf(size);
            return grown;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        internal static TrecsListHeader* GrowDetached(
            NativeHeapEntry* slot,
            PtrHandle handle,
            NativeHeapDetachLog* log,
            int minCapacity
        )
        {
            var header = (TrecsListHeader*)slot->Address.ToPointer();
            var size = GrowBlockSize(header->Capacity, minCapacity);
            var grown = (TrecsListHeader*)
                DetachedBlock.Reallocate(slot, handle, log, size, LiveBytes(header));
            grown->Capacity = CapacityOf(size);
            return grown;
        }
    }
}

namespace Trecs
{
    /// <summary>
    /// Static factories for <see cref="TrecsList{T}"/>. One heap slot holds the header and
    /// the elements, tagged <c>TypeId&lt;TrecsList&lt;T&gt;&gt;</c>.
    /// </summary>
    public static class TrecsList
    {
        /// <summary>
        /// Allocates a list with room for at least <paramref name="initialCapacity"/>
        /// elements. Capacity fills the heap slot, so 0 still gives room for a few small
        /// elements.
        /// </summary>
        public static TrecsList<T> Alloc<T>(
            WorldAccessor world,
            int initialCapacity = 0,
            [CallerFilePath] string callerFile = null,
            [CallerLineNumber] int callerLine = 0
        )
            where T : unmanaged
        {
            world.AssertCanMutateHeap();
            return Alloc<T>(world.NativeUniqueChunkStore, initialCapacity, callerFile, callerLine);
        }

        internal static unsafe TrecsList<T> Alloc<T>(
            NativeHeap heap,
            int initialCapacity = 0,
            [CallerFilePath] string callerFile = null,
            [CallerLineNumber] int callerLine = 0
        )
            where T : unmanaged
        {
            TrecsDebugAssert.That(initialCapacity >= 0, "initialCapacity must be non-negative");
            TrecsAssert.That(
                TrecsListLayout<T>.Alignment <= DetachedBlock.MaxAlignment,
                "TrecsList element alignment {0} exceeds {1}",
                TrecsListLayout<T>.Alignment,
                DetachedBlock.MaxAlignment
            );

            var size = TrecsListLayout<T>.BlockSizeFor(initialCapacity);
            var handle = heap.Alloc(
                size,
                TrecsListLayout<T>.Alignment,
                TypeId<TrecsList<T>>.Value.Value,
                out var address,
                callerFile,
                callerLine
            );
            ((TrecsListHeader*)address.ToPointer())->Capacity = TrecsListLayout<T>.CapacityOf(size);
            return new TrecsList<T>(handle);
        }
    }

    /// <summary>
    /// A growable list of unmanaged values on the world's <see cref="NativeHeap"/>, held by
    /// a 4-byte handle that never changes, so a component can carry it.
    ///
    /// <para>Views: managed <see cref="TrecsListRead{T}"/> / <see cref="TrecsListWrite{T}"/>
    /// (main thread) and Burst-safe <see cref="NativeTrecsListRead{T}"/> /
    /// <see cref="NativeTrecsListWrite{T}"/>. Every write view grows, in a job too: a job
    /// grows into a detached block that the heap moves back into its pages at the next phase
    /// boundary. <c>AddNoResize</c> skips the capacity branch.</para>
    ///
    /// <para><b>Write discipline.</b> <c>Write</c> and <c>EnsureCapacity</c> take
    /// <c>this ref TrecsList&lt;T&gt;</c>, so writing through an <c>in</c> parameter or an
    /// <c>IRead&lt;...&gt;</c> aspect field is a compile error.</para>
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public readonly struct TrecsList<T> : IEquatable<TrecsList<T>>
        where T : unmanaged
    {
        public readonly PtrHandle Handle;

        // Internal so external code can't fabricate a TrecsList<T> from an arbitrary uint.
        internal TrecsList(PtrHandle handle)
        {
            Handle = handle;
        }

        public readonly bool IsNull => Handle.IsNull;

        // Readonly so an OnRemoved observer, which gets the component as `in`, can dispose.
        public readonly void Dispose(WorldAccessor world)
        {
            world.AssertCanMutateHeap();
            Dispose(world.NativeUniqueChunkStore);
        }

        internal readonly unsafe void Dispose(NativeHeap heap)
        {
            ResolveSlot(in heap.Resolver, out _);
            heap.Free(Handle);
        }

        internal readonly unsafe void EnsureCapacity(NativeHeap heap, int minCapacity)
        {
            TrecsDebugAssert.That(minCapacity >= 0, "minCapacity must be non-negative");
            var header = (TrecsListHeader*)ResolveSlot(in heap.Resolver, out _).Address.ToPointer();
            if (minCapacity > header->Capacity)
            {
                TrecsListLayout<T>.Grow(heap, Handle, header, minCapacity);
            }
        }

        /// <summary>
        /// Opens a main-thread read view. In a job use <see cref="Read(in NativeWorldAccessor)"/>.
        /// </summary>
        public readonly TrecsListRead<T> Read(WorldAccessor world) =>
            Read(world.NativeUniqueChunkStore);

        internal readonly unsafe TrecsListRead<T> Read(NativeHeap heap)
        {
            var entry = ResolveSlot(in heap.Resolver, out var slot);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            return new TrecsListRead<T>(slot, Handle, entry.Safety);
#else
            return new TrecsListRead<T>(slot, Handle);
#endif
        }

        internal readonly unsafe TrecsListWrite<T> Write(NativeHeap heap)
        {
            var entry = ResolveSlot(in heap.Resolver, out var slot);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            return new TrecsListWrite<T>(slot, Handle, heap, entry.Safety);
#else
            return new TrecsListWrite<T>(slot, Handle, heap);
#endif
        }

        public readonly NativeTrecsListRead<T> Read(in NativeWorldAccessor nativeWorld) =>
            Read(nativeWorld.ChunkStoreResolver);

        internal readonly NativeTrecsListWrite<T> Write(in NativeWorldAccessor nativeWorld)
        {
            nativeWorld.AssertCanMutateHeap();
            return Write(nativeWorld.ChunkStoreResolver);
        }

        /// <summary>
        /// Burst-compatible read view. Pass a resolver from
        /// <see cref="WorldAccessor.NativeHeapResolver"/> or
        /// <see cref="NativeWorldAccessor.ChunkStoreResolver"/>. Throws if the handle is
        /// stale or not a <c>TrecsList&lt;T&gt;</c>.
        /// </summary>
        public readonly unsafe NativeTrecsListRead<T> Read(in NativeHeapResolver resolver)
        {
            var entry = ResolveSlot(in resolver, out _);
            var header = (TrecsListHeader*)entry.Address.ToPointer();
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            return new NativeTrecsListRead<T>(header, entry.Safety);
#else
            return new NativeTrecsListRead<T>(header);
#endif
        }

        // Public access goes through TrecsListExtensions.Write(this ref, ...) so the ref
        // discipline applies. The resolver carries the accessor's role flag.
        internal readonly unsafe NativeTrecsListWrite<T> Write(in NativeHeapResolver resolver)
        {
            resolver.AssertCanMutateHeap();
            var entry = ResolveSlot(in resolver, out var slot);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            return new NativeTrecsListWrite<T>(slot, Handle, resolver.DetachLog, entry.Safety);
#else
            return new NativeTrecsListWrite<T>(slot, Handle, resolver.DetachLog);
#endif
        }

        readonly unsafe NativeHeapEntry ResolveSlot(
            in NativeHeapResolver resolver,
            out NativeHeapEntry* slot
        )
        {
            var entry = resolver.ResolveEntryWithSlotPtr(Handle, out slot);
            TrecsAssert.That(
                entry.TypeHash == TypeId<TrecsList<T>>.Value.Value,
                "TrecsList type-hash mismatch for handle {0}: stored {1} != expected {2}",
                Handle.Value,
                entry.TypeHash,
                TypeId<TrecsList<T>>.Value.Value
            );
            return entry;
        }

        public readonly bool Equals(TrecsList<T> other) => Handle.Equals(other.Handle);

        public override readonly bool Equals(object obj) =>
            obj is TrecsList<T> other && Equals(other);

        public override readonly int GetHashCode() => Handle.GetHashCode();

        public static bool operator ==(TrecsList<T> left, TrecsList<T> right) => left.Equals(right);

        public static bool operator !=(TrecsList<T> left, TrecsList<T> right) =>
            !left.Equals(right);
    }

    /// <summary>
    /// Mutating operations on <see cref="TrecsList{T}"/>. Each takes
    /// <c>this ref TrecsList&lt;T&gt;</c>, so the caller must hold writable access to the
    /// handle. Same pattern as <see cref="FixedList16Extensions"/>.
    /// </summary>
    public static class TrecsListExtensions
    {
        public static TrecsListWrite<T> Write<T>(this ref TrecsList<T> list, WorldAccessor world)
            where T : unmanaged
        {
            world.AssertCanMutateHeap();
            return list.Write(world.NativeUniqueChunkStore);
        }

        public static NativeTrecsListWrite<T> Write<T>(
            this ref TrecsList<T> list,
            in NativeWorldAccessor nativeWorld
        )
            where T : unmanaged => list.Write(in nativeWorld);

        public static NativeTrecsListWrite<T> Write<T>(
            this ref TrecsList<T> list,
            in NativeHeapResolver resolver
        )
            where T : unmanaged => list.Write(in resolver);

        /// <summary>
        /// Grows the list to hold at least <paramref name="minCapacity"/> elements, doubling
        /// past the current capacity. Main thread; throws if a job still holds a view.
        /// </summary>
        public static void EnsureCapacity<T>(
            this ref TrecsList<T> list,
            WorldAccessor world,
            int minCapacity
        )
            where T : unmanaged
        {
            world.AssertCanMutateHeap();
            list.EnsureCapacity(world.NativeUniqueChunkStore, minCapacity);
        }
    }
}
