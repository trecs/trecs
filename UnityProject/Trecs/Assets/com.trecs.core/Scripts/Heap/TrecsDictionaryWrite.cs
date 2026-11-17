using System;
using System.Runtime.CompilerServices;
using Trecs.Internal;
using Unity.Collections.LowLevel.Unsafe;

namespace Trecs
{
    /// <summary>
    /// Main-thread writable view over a <see cref="TrecsDictionary{TKey, TValue}"/>. Adding
    /// grows in place in the heap; the <c>NoResize</c> variants skip the capacity branch.
    /// Reaches the block through the allocation's side-table entry on every access and
    /// asserts the allocation is still live. In a job use
    /// <see cref="NativeTrecsDictionaryWrite{TKey, TValue}"/>.
    /// </summary>
    public unsafe ref struct TrecsDictionaryWrite<TKey, TValue>
        where TKey : unmanaged, IEquatable<TKey>
        where TValue : unmanaged
    {
        readonly NativeHeapEntry* _slot;
        readonly PtrHandle _handle;
        readonly NativeHeap _heap;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        readonly AtomicSafetyHandle m_Safety;

        internal TrecsDictionaryWrite(
            NativeHeapEntry* slot,
            PtrHandle handle,
            NativeHeap heap,
            AtomicSafetyHandle safety
        )
        {
            _slot = slot;
            _handle = handle;
            _heap = heap;
            m_Safety = safety;
        }
#else
        internal TrecsDictionaryWrite(NativeHeapEntry* slot, PtrHandle handle, NativeHeap heap)
        {
            _slot = slot;
            _handle = handle;
            _heap = heap;
        }
#endif

        TrecsDictionaryHeader* ReadHeader
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                AtomicSafetyHandle.CheckReadAndThrow(m_Safety);
#endif
                NativeHeapResolver.AssertSlotLive(_slot, _handle);
                return (TrecsDictionaryHeader*)_slot->Address.ToPointer();
            }
        }

        TrecsDictionaryHeader* WriteHeader
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#endif
                NativeHeapResolver.AssertSlotLive(_slot, _handle);
                return (TrecsDictionaryHeader*)_slot->Address.ToPointer();
            }
        }

        public int Count
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ReadHeader->Count;
        }

        public int Capacity
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ReadHeader->EntryCapacity;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool ContainsKey(TKey key) =>
            TrecsDictionaryLayout<TKey, TValue>.TryGetIndex(ReadHeader, key, out _);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryGetValue(TKey key, out TValue value)
        {
            var header = ReadHeader;
            if (TrecsDictionaryLayout<TKey, TValue>.TryGetIndex(header, key, out var index))
            {
                value = TrecsDictionaryLayout<TKey, TValue>.Values(header)[index];
                return true;
            }
            value = default;
            return false;
        }

        public TValue this[TKey key]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                var header = ReadHeader;
                TrecsAssert.That(
                    TrecsDictionaryLayout<TKey, TValue>.TryGetIndex(header, key, out var index),
                    "Key not found in TrecsDictionary"
                );
                return TrecsDictionaryLayout<TKey, TValue>.Values(header)[index];
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => GetOrAdd(key) = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(TKey key, in TValue value)
        {
            var added = AddKey(key, out var header, out var index);
            TrecsDebugAssert.That(added, "Key already present in TrecsDictionary");
            TrecsDictionaryLayout<TKey, TValue>.Values(header)[index] = value;
        }

        /// <summary>Adds a missing key; the dictionary must have room.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddNoResize(TKey key, in TValue value)
        {
            var added = AddKeyNoResize(key, out var header, out var index);
            TrecsDebugAssert.That(added, "Key already present in TrecsDictionary");
            TrecsDictionaryLayout<TKey, TValue>.Values(header)[index] = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryAdd(TKey key, in TValue value)
        {
            if (!AddKey(key, out var header, out var index))
                return false;
            TrecsDictionaryLayout<TKey, TValue>.Values(header)[index] = value;
            return true;
        }

        /// <summary><see cref="TryAdd"/> that needs room when the key is missing.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryAddNoResize(TKey key, in TValue value)
        {
            if (!AddKeyNoResize(key, out var header, out var index))
                return false;
            TrecsDictionaryLayout<TKey, TValue>.Values(header)[index] = value;
            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Set(TKey key, in TValue value) => GetValueByRef(key) = value;

        /// <summary>The value of <paramref name="key"/>, added as default when missing.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref TValue GetOrAdd(TKey key)
        {
            AddKey(key, out var header, out var index);
            return ref TrecsDictionaryLayout<TKey, TValue>.Values(header)[index];
        }

        /// <summary><see cref="GetOrAdd"/> that needs room when the key is missing.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref TValue GetOrAddNoResize(TKey key)
        {
            AddKeyNoResize(key, out var header, out var index);
            return ref TrecsDictionaryLayout<TKey, TValue>.Values(header)[index];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref TValue GetValueByRef(TKey key)
        {
            var header = WriteHeader;
            TrecsAssert.That(
                TrecsDictionaryLayout<TKey, TValue>.TryGetIndex(header, key, out var index),
                "Key not found in TrecsDictionary"
            );
            return ref TrecsDictionaryLayout<TKey, TValue>.Values(header)[index];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref TValue GetValueAtIndex(int index)
        {
            var header = WriteHeader;
            TrecsDebugAssert.That(
                (uint)index < (uint)header->Count,
                "TrecsDictionaryWrite.GetValueAtIndex index {0} out of range (Count={1})",
                index,
                header->Count
            );
            return ref TrecsDictionaryLayout<TKey, TValue>.Values(header)[index];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TKey GetKeyAtIndex(int index)
        {
            var header = ReadHeader;
            TrecsDebugAssert.That(
                (uint)index < (uint)header->Count,
                "TrecsDictionaryWrite.GetKeyAtIndex index {0} out of range (Count={1})",
                index,
                header->Count
            );
            return TrecsDictionaryLayout<TKey, TValue>.Nodes(header)[index].Key;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryGetIndex(TKey key, out int index) =>
            TrecsDictionaryLayout<TKey, TValue>.TryGetIndex(ReadHeader, key, out index);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Remove(TKey key) =>
            TrecsDictionaryLayout<TKey, TValue>.Remove(WriteHeader, key, out _);

        public bool Remove(TKey key, out TValue removedValue) =>
            TrecsDictionaryLayout<TKey, TValue>.Remove(WriteHeader, key, out removedValue);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Clear() => TrecsDictionaryLayout<TKey, TValue>.Clear(WriteHeader);

        /// <summary>
        /// Grows the dictionary to hold at least <paramref name="minCapacity"/> entries,
        /// doubling past the current capacity.
        /// </summary>
        public void EnsureCapacity(int minCapacity)
        {
            TrecsDebugAssert.That(minCapacity >= 0, "minCapacity must be non-negative");
            var header = WriteHeader;
            if (minCapacity > header->EntryCapacity)
            {
                TrecsDictionaryLayout<TKey, TValue>.Grow(_heap, _handle, header, minCapacity);
            }
        }

        public TrecsDictionaryKeys<TKey> Keys
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                var header = ReadHeader;
                return new TrecsDictionaryKeys<TKey>(
                    TrecsDictionaryLayout<TKey, TValue>.Nodes(header),
                    header->Count
                );
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TrecsDictionaryEnumerator<TKey, TValue> GetEnumerator()
        {
            var header = WriteHeader;
            return new TrecsDictionaryEnumerator<TKey, TValue>(
                TrecsDictionaryLayout<TKey, TValue>.Nodes(header),
                TrecsDictionaryLayout<TKey, TValue>.Values(header),
                header->Count
            );
        }

        // False with the key's index when present; else appends it, growing when full.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        bool AddKey(in TKey key, out TrecsDictionaryHeader* header, out int index)
        {
            header = WriteHeader;
            var hash = key.GetHashCode();
            if (TrecsDictionaryLayout<TKey, TValue>.TryFind(header, key, hash, out index))
                return false;
            if (header->Count == header->EntryCapacity)
            {
                header = TrecsDictionaryLayout<TKey, TValue>.Grow(
                    _heap,
                    _handle,
                    header,
                    header->Count + 1
                );
            }
            index = TrecsDictionaryLayout<TKey, TValue>.AppendKey(header, key, hash);
            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        bool AddKeyNoResize(in TKey key, out TrecsDictionaryHeader* header, out int index)
        {
            header = WriteHeader;
            var hash = key.GetHashCode();
            if (TrecsDictionaryLayout<TKey, TValue>.TryFind(header, key, hash, out index))
                return false;
            index = TrecsDictionaryLayout<TKey, TValue>.AppendKey(header, key, hash);
            return true;
        }
    }
}
