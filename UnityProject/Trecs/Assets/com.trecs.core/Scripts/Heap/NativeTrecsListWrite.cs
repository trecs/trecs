using System.Runtime.CompilerServices;
using Trecs.Internal;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Trecs
{
    /// <summary>
    /// Burst-safe writable view over a <see cref="TrecsList{T}"/>. <see cref="Add"/> and
    /// <see cref="EnsureCapacity"/> grow, on any thread: the list moves to a detached block
    /// that the heap moves back into its pages at the next phase boundary.
    /// <see cref="AddNoResize"/> skips the capacity branch.
    ///
    /// <para>Reaches the block through the allocation's side-table entry on every access, so
    /// copies of the view stay valid across a grow. Exclusive: a second view over the same
    /// list is rejected at <c>Schedule</c>.</para>
    /// </summary>
    [NativeContainer]
    public readonly unsafe struct NativeTrecsListWrite<T>
        where T : unmanaged
    {
        [NativeDisableUnsafePtrRestriction]
        readonly NativeHeapEntry* _slot;

        [NativeDisableUnsafePtrRestriction]
        readonly NativeHeapDetachLog* _detachLog;

        readonly PtrHandle _handle;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        readonly AtomicSafetyHandle m_Safety;

        static readonly SharedStatic<int> s_staticSafetyId = SharedStatic<int>.GetOrCreate<
            NativeTrecsListWrite<T>
        >();

        internal NativeTrecsListWrite(
            NativeHeapEntry* slot,
            PtrHandle handle,
            NativeHeapDetachLog* detachLog,
            AtomicSafetyHandle safety
        )
        {
            _slot = slot;
            _handle = handle;
            _detachLog = detachLog;
            m_Safety = safety;
            CollectionHelper.SetStaticSafetyId<NativeTrecsListWrite<T>>(
                ref m_Safety,
                ref s_staticSafetyId.Data
            );
        }
#else
        internal NativeTrecsListWrite(
            NativeHeapEntry* slot,
            PtrHandle handle,
            NativeHeapDetachLog* detachLog
        )
        {
            _slot = slot;
            _handle = handle;
            _detachLog = detachLog;
        }
#endif

        TrecsListHeader* ReadHeader
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                AtomicSafetyHandle.CheckReadAndThrow(m_Safety);
#endif
                return (TrecsListHeader*)_slot->Address.ToPointer();
            }
        }

        TrecsListHeader* WriteHeader
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#endif
                return (TrecsListHeader*)_slot->Address.ToPointer();
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
            get => ReadHeader->Capacity;
        }

        public ref T this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                var header = WriteHeader;
                TrecsDebugAssert.That(
                    (uint)index < (uint)header->Count,
                    "NativeTrecsListWrite index {0} out of range (Count={1})",
                    index,
                    header->Count
                );
                return ref TrecsListLayout<T>.Data(header)[index];
            }
        }

        /// <summary>Appends <paramref name="value"/>, doubling the capacity when full.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(in T value)
        {
            var header = WriteHeader;
            var count = header->Count;
            if (count == header->Capacity)
            {
                header = TrecsListLayout<T>.GrowDetached(_slot, _handle, _detachLog, count + 1);
            }
            TrecsListLayout<T>.Data(header)[count] = value;
            header->Count = count + 1;
        }

        /// <summary>Appends <paramref name="value"/>; the list must have room.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddNoResize(in T value)
        {
            var header = WriteHeader;
            var count = header->Count;
            TrecsDebugAssert.That(
                count < header->Capacity,
                "NativeTrecsListWrite.AddNoResize: list is full (capacity {0})",
                header->Capacity
            );
            TrecsListLayout<T>.Data(header)[count] = value;
            header->Count = count + 1;
        }

        /// <summary>
        /// Inserts <paramref name="value"/> at <paramref name="index"/>, shifting later
        /// elements up (O(N)), doubling the capacity when full.
        /// </summary>
        public void Insert(int index, in T value)
        {
            var header = WriteHeader;
            if (header->Count == header->Capacity)
            {
                header = TrecsListLayout<T>.GrowDetached(
                    _slot,
                    _handle,
                    _detachLog,
                    header->Count + 1
                );
            }
            TrecsListLayout<T>.InsertNoResize(header, index, value);
        }

        /// <summary><see cref="Insert"/>; the list must have room.</summary>
        public void InsertNoResize(int index, in T value) =>
            TrecsListLayout<T>.InsertNoResize(WriteHeader, index, value);

        /// <summary>
        /// Grows the list to hold at least <paramref name="minCapacity"/> elements, doubling
        /// past the current capacity.
        /// </summary>
        public void EnsureCapacity(int minCapacity)
        {
            TrecsDebugAssert.That(minCapacity >= 0, "minCapacity must be non-negative");
            if (minCapacity > WriteHeader->Capacity)
            {
                TrecsListLayout<T>.GrowDetached(_slot, _handle, _detachLog, minCapacity);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Clear()
        {
            WriteHeader->Count = 0;
        }

        /// <summary>Removes element at <paramref name="index"/>, shifting later elements down (O(N)).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void RemoveAt(int index)
        {
            var header = WriteHeader;
            TrecsDebugAssert.That(
                (uint)index < (uint)header->Count,
                "NativeTrecsListWrite.RemoveAt index {0} out of range (Count={1})",
                index,
                header->Count
            );
            var data = TrecsListLayout<T>.Data(header);
            var tail = header->Count - index - 1;
            if (tail > 0)
            {
                UnsafeUtility.MemMove(data + index, data + index + 1, (long)tail * sizeof(T));
            }
            header->Count -= 1;
        }

        /// <summary>
        /// Removes the element at <paramref name="index"/> by swapping the last element into
        /// its slot (O(1), does not preserve order).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void RemoveAtSwapBack(int index)
        {
            var header = WriteHeader;
            TrecsDebugAssert.That(
                (uint)index < (uint)header->Count,
                "NativeTrecsListWrite.RemoveAtSwapBack index {0} out of range (Count={1})",
                index,
                header->Count
            );
            var data = TrecsListLayout<T>.Data(header);
            var last = header->Count - 1;
            if (index != last)
            {
                data[index] = data[last];
            }
            header->Count = last;
        }
    }
}
