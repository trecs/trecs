using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Trecs.Internal;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Trecs
{
    /// <summary>
    /// Main-thread writable view over a <see cref="TrecsList{T}"/>. <see cref="Add"/> grows
    /// in place in the heap; <see cref="AddNoResize"/> skips the capacity branch. In a job
    /// use <see cref="NativeTrecsListWrite{T}"/>.
    ///
    /// <para>Reaches the block through the allocation's side-table entry on every access, so
    /// sibling views stay valid across a grow, and asserts the allocation is still live
    /// (shipping builds too). A <c>ref struct</c>: it cannot outlive the method that opened
    /// it.</para>
    /// </summary>
    public unsafe ref struct TrecsListWrite<T>
        where T : unmanaged
    {
        readonly NativeHeapEntry* _slot;
        readonly PtrHandle _handle;
        readonly NativeHeap _heap;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        readonly AtomicSafetyHandle m_Safety;

        internal TrecsListWrite(
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
        internal TrecsListWrite(NativeHeapEntry* slot, PtrHandle handle, NativeHeap heap)
        {
            _slot = slot;
            _handle = handle;
            _heap = heap;
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
                NativeHeapResolver.AssertSlotLive(_slot, _handle);
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
                NativeHeapResolver.AssertSlotLive(_slot, _handle);
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
                    "TrecsListWrite index {0} out of range (Count={1})",
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
                header = TrecsListLayout<T>.Grow(_heap, _handle, header, count + 1);
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
                "TrecsListWrite.AddNoResize: list is full (capacity {0})",
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
                header = TrecsListLayout<T>.Grow(_heap, _handle, header, header->Count + 1);
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
            var header = WriteHeader;
            if (minCapacity > header->Capacity)
            {
                TrecsListLayout<T>.Grow(_heap, _handle, header, minCapacity);
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
                "TrecsListWrite.RemoveAt index {0} out of range (Count={1})",
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
                "TrecsListWrite.RemoveAtSwapBack index {0} out of range (Count={1})",
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

        /// <summary>Sorts the list in place with <paramref name="comparer"/> (O(N log N)).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Sort<TComparer>(TComparer comparer)
            where TComparer : IComparer<T>
        {
            var header = WriteHeader;
            NativeSortExtension.Sort(TrecsListLayout<T>.Data(header), header->Count, comparer);
        }

        /// <summary>Enables <c>foreach (ref var x in w)</c>; elements may be changed in place.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TrecsListWriteEnumerator<T> GetEnumerator() => new TrecsListWriteEnumerator<T>(this);
    }

    public ref struct TrecsListWriteEnumerator<T>
        where T : unmanaged
    {
        TrecsListWrite<T> _view;
        int _index;

        public TrecsListWriteEnumerator(TrecsListWrite<T> view)
        {
            _view = view;
            _index = -1;
        }

        public ref T Current
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _view[_index];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool MoveNext()
        {
            _index++;
            return _index < _view.Count;
        }
    }

    /// <summary>
    /// Default-ordering sort for <see cref="TrecsListWrite{T}"/>: an extension because a
    /// member cannot add <see cref="IComparable{T}"/> to <c>T</c>.
    /// </summary>
    public static class TrecsListWriteSortExtensions
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Sort<T>(this ref TrecsListWrite<T> list)
            where T : unmanaged, IComparable<T>
        {
            list.Sort(default(DefaultComparer<T>));
        }

        struct DefaultComparer<T> : IComparer<T>
            where T : IComparable<T>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public int Compare(T x, T y) => x.CompareTo(y);
        }
    }
}
