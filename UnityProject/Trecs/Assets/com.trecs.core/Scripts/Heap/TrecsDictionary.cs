using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Trecs.Collections;
using Trecs.Internal;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Trecs.Internal
{
    /// <summary>Front of a <see cref="TrecsDictionary{TKey, TValue}"/> block.</summary>
    [EditorBrowsable(EditorBrowsableState.Never)]
    public struct TrecsDictionaryHeader
    {
        public ulong FastModMultiplier;
        public int Count;
        public int EntryCapacity;
        public int BucketCount;
    }

    /// <summary>
    /// Block layout of a <see cref="TrecsDictionary{TKey, TValue}"/>: header, nodes, values,
    /// buckets (1-based index of a chain's last node, 0 when empty). Capacity fills the heap
    /// slot; the bucket count is the largest prime ≤ capacity.
    /// </summary>
    [EditorBrowsable(EditorBrowsableState.Never)]
    public static unsafe class TrecsDictionaryLayout<TKey, TValue>
        where TKey : unmanaged, IEquatable<TKey>
        where TValue : unmanaged
    {
        static int NodeSize => UnsafeUtility.SizeOf<IterableDictionaryNode<TKey>>();
        static int NodeAlign => UnsafeUtility.AlignOf<IterableDictionaryNode<TKey>>();
        static int ValueAlign => UnsafeUtility.AlignOf<TValue>();
        static int NodesOffset => AlignUp(sizeof(TrecsDictionaryHeader), NodeAlign);
        static int EntryBytes => NodeSize + sizeof(TValue) + sizeof(int);

        // Alignment padding before the values and before the buckets.
        static int Padding => ValueAlign - 1 + sizeof(int) - 1;

        public static int Alignment =>
            math.max(
                UnsafeUtility.AlignOf<TrecsDictionaryHeader>(),
                math.max(NodeAlign, ValueAlign)
            );

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int AlignUp(int value, int alignment) => (value + alignment - 1) & -alignment;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int ValuesOffset(int capacity) =>
            AlignUp(NodesOffset + capacity * NodeSize, ValueAlign);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int BucketsOffset(int capacity) =>
            AlignUp(ValuesOffset(capacity) + capacity * sizeof(TValue), sizeof(int));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IterableDictionaryNode<TKey>* Nodes(TrecsDictionaryHeader* header) =>
            (IterableDictionaryNode<TKey>*)((byte*)header + NodesOffset);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TValue* Values(TrecsDictionaryHeader* header) =>
            (TValue*)((byte*)header + ValuesOffset(header->EntryCapacity));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int* Buckets(TrecsDictionaryHeader* header) =>
            (int*)((byte*)header + BucketsOffset(header->EntryCapacity));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static ref int Bucket(TrecsDictionaryHeader* header, int hash) =>
            ref Buckets(header)[
                TrecsDictionary.Reduce(
                    (uint)hash,
                    (uint)header->BucketCount,
                    header->FastModMultiplier
                )
            ];

        internal static int BlockSizeFor(int minCapacity)
        {
            var bytes = NodesOffset + (long)minCapacity * EntryBytes + Padding;
            TrecsAssert.That(
                bytes <= int.MaxValue,
                "TrecsDictionary of {0} entries of {1} bytes exceeds int.MaxValue bytes",
                minCapacity,
                EntryBytes
            );
            return NativeHeapResolver.GetSlotByteSize((int)bytes, Alignment);
        }

        // Doubles the capacity, or takes minCapacity when doubling would pass int.MaxValue bytes.
        static int GrowBlockSize(int capacity, int minCapacity)
        {
            var doubled = math.max((long)minCapacity, 2L * capacity);
            var fits = NodesOffset + doubled * EntryBytes + Padding <= int.MaxValue;
            return BlockSizeFor(fits ? (int)doubled : minCapacity);
        }

        internal static void SetCapacity(TrecsDictionaryHeader* header, int blockSize)
        {
            var capacity = math.max(0, (blockSize - NodesOffset - Padding) / EntryBytes);
            var buckets = HashHelpers.GetPrimeAtMost(capacity);
            header->EntryCapacity = capacity;
            header->BucketCount = buckets;
            header->FastModMultiplier =
                buckets > 0 ? HashHelpers.GetFastModMultiplier((uint)buckets) : 0;
        }

        static int LiveBytes(TrecsDictionaryHeader* header) =>
            ValuesOffset(header->EntryCapacity) + header->Count * sizeof(TValue);

        // Lays out a grown block that holds a copy of the old block's first LiveBytes.
        static void Relayout(TrecsDictionaryHeader* header, int oldCapacity, int blockSize)
        {
            var block = (byte*)header;
            var count = header->Count;
            var oldValues = block + ValuesOffset(oldCapacity);
            SetCapacity(header, blockSize);
            var values = (byte*)Values(header);
            UnsafeUtility.MemMove(values, oldValues, (long)count * sizeof(TValue));
            var liveNodesEnd = NodesOffset + count * NodeSize;
            UnsafeUtility.MemClear(block + liveNodesEnd, values - block - liveNodesEnd);

            var nodes = Nodes(header);
            for (int i = 0; i < count; i++)
            {
                ref var bucket = ref Bucket(header, nodes[i].HashCode);
                nodes[i].Previous = bucket - 1;
                bucket = i + 1;
            }
        }

        internal static TrecsDictionaryHeader* Grow(
            NativeHeap heap,
            PtrHandle handle,
            TrecsDictionaryHeader* header,
            int minCapacity
        )
        {
            var oldCapacity = header->EntryCapacity;
            var size = GrowBlockSize(oldCapacity, minCapacity);
            var grown = (TrecsDictionaryHeader*)
                heap.Reallocate(handle, size, Alignment, LiveBytes(header));
            Relayout(grown, oldCapacity, size);
            return grown;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        internal static TrecsDictionaryHeader* GrowDetached(
            NativeHeapEntry* slot,
            PtrHandle handle,
            NativeHeapDetachLog* log,
            int minCapacity
        )
        {
            var header = (TrecsDictionaryHeader*)slot->Address.ToPointer();
            var oldCapacity = header->EntryCapacity;
            var size = GrowBlockSize(oldCapacity, minCapacity);
            var grown = (TrecsDictionaryHeader*)
                DetachedBlock.Reallocate(slot, handle, log, size, LiveBytes(header));
            Relayout(grown, oldCapacity, size);
            return grown;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool TryGetIndex(TrecsDictionaryHeader* header, in TKey key, out int index) =>
            TryFind(header, key, key.GetHashCode(), out index);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool TryFind(
            TrecsDictionaryHeader* header,
            in TKey key,
            int hash,
            out int index
        )
        {
            if (header->BucketCount != 0)
            {
                var nodes = Nodes(header);
                index = Bucket(header, hash) - 1;
                while (index != -1)
                {
                    ref var node = ref nodes[index];
                    if (node.HashCode == hash && node.Key.Equals(key))
                    {
                        return true;
                    }
                    index = node.Previous;
                }
            }
            index = 0;
            return false;
        }

        /// <summary>Appends a key known to be absent; the dictionary must have room.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int AppendKey(TrecsDictionaryHeader* header, in TKey key, int hash)
        {
            var index = header->Count;
            TrecsDebugAssert.That(
                index < header->EntryCapacity,
                "TrecsDictionary is full (capacity {0})",
                header->EntryCapacity
            );
            ref var bucket = ref Bucket(header, hash);
            ref var node = ref Nodes(header)[index];
            node.Key = key;
            node.HashCode = hash;
            node.Previous = bucket - 1;
            bucket = index + 1;
            Values(header)[index] = default;
            header->Count = index + 1;
            return index;
        }

        // Swap-back removal: the last entry moves into the freed index.
        internal static bool Remove(
            TrecsDictionaryHeader* header,
            in TKey key,
            out TValue removedValue
        )
        {
            if (header->BucketCount == 0)
            {
                removedValue = default;
                return false;
            }

            var nodes = Nodes(header);
            var values = Values(header);
            var buckets = Buckets(header);
            var bucketCount = (uint)header->BucketCount;
            var fastMod = header->FastModMultiplier;
            int hash = key.GetHashCode();
            int bucketIndex = TrecsDictionary.Reduce((uint)hash, bucketCount, fastMod);

            int indexToRemove = buckets[bucketIndex] - 1;
            int itemAfterCurrent = -1;
            while (indexToRemove != -1)
            {
                ref var node = ref nodes[indexToRemove];
                if (node.HashCode == hash && node.Key.Equals(key))
                {
                    if (buckets[bucketIndex] - 1 == indexToRemove)
                    {
                        buckets[bucketIndex] = node.Previous + 1;
                    }
                    else
                    {
                        nodes[itemAfterCurrent].Previous = node.Previous;
                    }
                    break;
                }
                itemAfterCurrent = indexToRemove;
                indexToRemove = node.Previous;
            }

            if (indexToRemove == -1)
            {
                removedValue = default;
                return false;
            }

            removedValue = values[indexToRemove];
            var lastIndex = header->Count - 1;
            if (indexToRemove != lastIndex)
            {
                ref var nodeToMove = ref nodes[lastIndex];
                var movingBucketIndex = TrecsDictionary.Reduce(
                    (uint)nodeToMove.HashCode,
                    bucketCount,
                    fastMod
                );

                var linkedListIter = buckets[movingBucketIndex] - 1;
                if (linkedListIter == lastIndex)
                {
                    buckets[movingBucketIndex] = indexToRemove + 1;
                }
                while (
                    nodes[linkedListIter].Previous != -1
                    && nodes[linkedListIter].Previous != lastIndex
                )
                {
                    linkedListIter = nodes[linkedListIter].Previous;
                }
                if (nodes[linkedListIter].Previous != -1)
                {
                    nodes[linkedListIter].Previous = indexToRemove;
                }

                nodes[indexToRemove] = nodeToMove;
                values[indexToRemove] = values[lastIndex];
            }

            header->Count = lastIndex;
            return true;
        }

        internal static void Clear(TrecsDictionaryHeader* header)
        {
            if (header->Count == 0)
            {
                return;
            }
            header->Count = 0;
            UnsafeUtility.MemClear(Buckets(header), header->BucketCount * sizeof(int));
        }
    }
}

namespace Trecs
{
    public static class TrecsDictionary
    {
        /// <summary>
        /// Allocates a dictionary with room for at least <paramref name="initialCapacity"/>
        /// entries. Capacity fills the heap slot.
        /// </summary>
        public static TrecsDictionary<TKey, TValue> Alloc<TKey, TValue>(
            WorldAccessor world,
            int initialCapacity = 0,
            [CallerFilePath] string callerFile = null,
            [CallerLineNumber] int callerLine = 0
        )
            where TKey : unmanaged, IEquatable<TKey>
            where TValue : unmanaged
        {
            world.AssertCanMutateHeap();
            return Alloc<TKey, TValue>(
                world.NativeUniqueChunkStore,
                initialCapacity,
                callerFile,
                callerLine
            );
        }

        internal static unsafe TrecsDictionary<TKey, TValue> Alloc<TKey, TValue>(
            NativeHeap heap,
            int initialCapacity = 0,
            [CallerFilePath] string callerFile = null,
            [CallerLineNumber] int callerLine = 0
        )
            where TKey : unmanaged, IEquatable<TKey>
            where TValue : unmanaged
        {
            TrecsDebugAssert.That(initialCapacity >= 0, "initialCapacity must be non-negative");
            var alignment = TrecsDictionaryLayout<TKey, TValue>.Alignment;
            TrecsAssert.That(
                alignment <= DetachedBlock.MaxAlignment,
                "TrecsDictionary entry alignment {0} exceeds {1}",
                alignment,
                DetachedBlock.MaxAlignment
            );

            var size = TrecsDictionaryLayout<TKey, TValue>.BlockSizeFor(initialCapacity);
            var handle = heap.Alloc(
                size,
                alignment,
                TypeId<TrecsDictionary<TKey, TValue>>.Value.Value,
                out var address,
                callerFile,
                callerLine
            );
            TrecsDictionaryLayout<TKey, TValue>.SetCapacity(
                (TrecsDictionaryHeader*)address.ToPointer(),
                size
            );
            return new TrecsDictionary<TKey, TValue>(handle);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int Reduce(uint hashcode, uint N, ulong fastModBucketsMultiplier)
        {
            if (hashcode >= N)
                return (int)HashHelpers.FastMod(hashcode, N, fastModBucketsMultiplier);
            return (int)hashcode;
        }
    }

    /// <summary>
    /// A growable hash dictionary of unmanaged keys and values on the world's
    /// <see cref="NativeHeap"/>, held by a 4-byte handle that never changes. Every write view
    /// grows, in a job too; <c>AddNoResize</c> and siblings skip the capacity branch. See
    /// <see cref="TrecsList{T}"/> for the view and growth model.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public readonly struct TrecsDictionary<TKey, TValue> : IEquatable<TrecsDictionary<TKey, TValue>>
        where TKey : unmanaged, IEquatable<TKey>
        where TValue : unmanaged
    {
        public readonly PtrHandle Handle;

        internal TrecsDictionary(PtrHandle handle)
        {
            Handle = handle;
        }

        public readonly bool IsNull => Handle.IsNull;

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
            var header = (TrecsDictionaryHeader*)
                ResolveSlot(in heap.Resolver, out _).Address.ToPointer();
            if (minCapacity > header->EntryCapacity)
            {
                TrecsDictionaryLayout<TKey, TValue>.Grow(heap, Handle, header, minCapacity);
            }
        }

        public readonly TrecsDictionaryRead<TKey, TValue> Read(WorldAccessor world) =>
            Read(world.NativeUniqueChunkStore);

        internal readonly unsafe TrecsDictionaryRead<TKey, TValue> Read(NativeHeap heap)
        {
            var entry = ResolveSlot(in heap.Resolver, out var slot);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            return new TrecsDictionaryRead<TKey, TValue>(slot, Handle, entry.Safety);
#else
            return new TrecsDictionaryRead<TKey, TValue>(slot, Handle);
#endif
        }

        internal readonly unsafe TrecsDictionaryWrite<TKey, TValue> Write(NativeHeap heap)
        {
            var entry = ResolveSlot(in heap.Resolver, out var slot);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            return new TrecsDictionaryWrite<TKey, TValue>(slot, Handle, heap, entry.Safety);
#else
            return new TrecsDictionaryWrite<TKey, TValue>(slot, Handle, heap);
#endif
        }

        public readonly NativeTrecsDictionaryRead<TKey, TValue> Read(
            in NativeWorldAccessor nativeWorld
        ) => Read(nativeWorld.ChunkStoreResolver);

        internal readonly NativeTrecsDictionaryWrite<TKey, TValue> Write(
            in NativeWorldAccessor nativeWorld
        )
        {
            nativeWorld.AssertCanMutateHeap();
            return Write(nativeWorld.ChunkStoreResolver);
        }

        public readonly unsafe NativeTrecsDictionaryRead<TKey, TValue> Read(
            in NativeHeapResolver resolver
        )
        {
            var entry = ResolveSlot(in resolver, out _);
            var header = (TrecsDictionaryHeader*)entry.Address.ToPointer();
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            return new NativeTrecsDictionaryRead<TKey, TValue>(header, entry.Safety);
#else
            return new NativeTrecsDictionaryRead<TKey, TValue>(header);
#endif
        }

        internal readonly unsafe NativeTrecsDictionaryWrite<TKey, TValue> Write(
            in NativeHeapResolver resolver
        )
        {
            resolver.AssertCanMutateHeap();
            var entry = ResolveSlot(in resolver, out var slot);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            return new NativeTrecsDictionaryWrite<TKey, TValue>(
                slot,
                Handle,
                resolver.DetachLog,
                entry.Safety
            );
#else
            return new NativeTrecsDictionaryWrite<TKey, TValue>(slot, Handle, resolver.DetachLog);
#endif
        }

        readonly unsafe NativeHeapEntry ResolveSlot(
            in NativeHeapResolver resolver,
            out NativeHeapEntry* slot
        )
        {
            var entry = resolver.ResolveEntryWithSlotPtr(Handle, out slot);
            TrecsAssert.That(
                entry.TypeHash == TypeId<TrecsDictionary<TKey, TValue>>.Value.Value,
                "TrecsDictionary type-hash mismatch for handle {0}: stored {1} != expected {2}",
                Handle.Value,
                entry.TypeHash,
                TypeId<TrecsDictionary<TKey, TValue>>.Value.Value
            );
            return entry;
        }

        public readonly bool Equals(TrecsDictionary<TKey, TValue> other) =>
            Handle.Equals(other.Handle);

        public override readonly bool Equals(object obj) =>
            obj is TrecsDictionary<TKey, TValue> other && Equals(other);

        public override readonly int GetHashCode() => Handle.GetHashCode();

        public static bool operator ==(
            TrecsDictionary<TKey, TValue> left,
            TrecsDictionary<TKey, TValue> right
        ) => left.Equals(right);

        public static bool operator !=(
            TrecsDictionary<TKey, TValue> left,
            TrecsDictionary<TKey, TValue> right
        ) => !left.Equals(right);
    }

    public static class TrecsDictionaryExtensions
    {
        public static TrecsDictionaryWrite<TKey, TValue> Write<TKey, TValue>(
            this ref TrecsDictionary<TKey, TValue> dict,
            WorldAccessor world
        )
            where TKey : unmanaged, IEquatable<TKey>
            where TValue : unmanaged
        {
            world.AssertCanMutateHeap();
            return dict.Write(world.NativeUniqueChunkStore);
        }

        public static NativeTrecsDictionaryWrite<TKey, TValue> Write<TKey, TValue>(
            this ref TrecsDictionary<TKey, TValue> dict,
            in NativeWorldAccessor nativeWorld
        )
            where TKey : unmanaged, IEquatable<TKey>
            where TValue : unmanaged => dict.Write(in nativeWorld);

        public static NativeTrecsDictionaryWrite<TKey, TValue> Write<TKey, TValue>(
            this ref TrecsDictionary<TKey, TValue> dict,
            in NativeHeapResolver resolver
        )
            where TKey : unmanaged, IEquatable<TKey>
            where TValue : unmanaged => dict.Write(in resolver);

        /// <summary>
        /// Grows the dictionary to hold at least <paramref name="minCapacity"/> entries,
        /// doubling past the current capacity. Main thread; throws if a job still holds a view.
        /// </summary>
        public static void EnsureCapacity<TKey, TValue>(
            this ref TrecsDictionary<TKey, TValue> dict,
            WorldAccessor world,
            int minCapacity
        )
            where TKey : unmanaged, IEquatable<TKey>
            where TValue : unmanaged
        {
            world.AssertCanMutateHeap();
            dict.EnsureCapacity(world.NativeUniqueChunkStore, minCapacity);
        }
    }

    /// <summary>Keys of a dictionary view in index order, valid until the dictionary changes.</summary>
    public unsafe struct TrecsDictionaryKeys<TKey>
        where TKey : unmanaged, IEquatable<TKey>
    {
        [NativeDisableUnsafePtrRestriction]
        readonly IterableDictionaryNode<TKey>* _nodes;
        readonly int _count;
        int _index;

        internal TrecsDictionaryKeys(IterableDictionaryNode<TKey>* nodes, int count)
        {
            _nodes = nodes;
            _count = count;
            _index = -1;
        }

        public TrecsDictionaryKeys<TKey> GetEnumerator() => this;

        public TKey Current
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _nodes[_index].Key;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool MoveNext() => ++_index < _count;
    }

    /// <summary>Entries of a dictionary view in index order, valid until the dictionary changes.</summary>
    public unsafe struct TrecsDictionaryEnumerator<TKey, TValue>
        where TKey : unmanaged, IEquatable<TKey>
        where TValue : unmanaged
    {
        [NativeDisableUnsafePtrRestriction]
        readonly IterableDictionaryNode<TKey>* _nodes;

        [NativeDisableUnsafePtrRestriction]
        readonly TValue* _values;
        readonly int _count;
        int _index;

        internal TrecsDictionaryEnumerator(
            IterableDictionaryNode<TKey>* nodes,
            TValue* values,
            int count
        )
        {
            _nodes = nodes;
            _values = values;
            _count = count;
            _index = -1;
        }

        public TrecsDictionaryEntry<TKey, TValue> Current
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => new TrecsDictionaryEntry<TKey, TValue>(_nodes[_index].Key, _values + _index);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool MoveNext() => ++_index < _count;
    }

    public readonly unsafe struct TrecsDictionaryEntry<TKey, TValue>
        where TValue : unmanaged
    {
        public readonly TKey Key;

        [NativeDisableUnsafePtrRestriction]
        readonly TValue* _value;

        internal TrecsDictionaryEntry(TKey key, TValue* value)
        {
            Key = key;
            _value = value;
        }

        public ref TValue Value
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref *_value;
        }

        public void Deconstruct(out TKey key, out TValue value)
        {
            key = Key;
            value = *_value;
        }
    }

    /// <summary>Read-only twin of <see cref="TrecsDictionaryEnumerator{TKey, TValue}"/>.</summary>
    public struct TrecsDictionaryReadEnumerator<TKey, TValue>
        where TKey : unmanaged, IEquatable<TKey>
        where TValue : unmanaged
    {
        TrecsDictionaryEnumerator<TKey, TValue> _inner;

        internal unsafe TrecsDictionaryReadEnumerator(
            IterableDictionaryNode<TKey>* nodes,
            TValue* values,
            int count
        )
        {
            _inner = new TrecsDictionaryEnumerator<TKey, TValue>(nodes, values, count);
        }

        public TrecsDictionaryReadEntry<TKey, TValue> Current
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => new TrecsDictionaryReadEntry<TKey, TValue>(_inner.Current);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool MoveNext() => _inner.MoveNext();
    }

    public readonly struct TrecsDictionaryReadEntry<TKey, TValue>
        where TValue : unmanaged
    {
        readonly TrecsDictionaryEntry<TKey, TValue> _inner;

        internal TrecsDictionaryReadEntry(TrecsDictionaryEntry<TKey, TValue> inner)
        {
            _inner = inner;
        }

        public TKey Key => _inner.Key;

        public ref readonly TValue Value
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _inner.Value;
        }

        public void Deconstruct(out TKey key, out TValue value) =>
            _inner.Deconstruct(out key, out value);
    }
}
