using System.Runtime.CompilerServices;
using Trecs.Internal;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Trecs
{
    /// <summary>
    /// Burst-safe read-only view over a <see cref="TrecsList{T}"/>. Many readers may hold it
    /// at once; a concurrent <see cref="NativeTrecsListWrite{T}"/> is rejected at
    /// <c>Schedule</c>. Caches the block pointer: nothing can grow the list while a reader
    /// holds it.
    /// </summary>
    [NativeContainer]
    [NativeContainerIsReadOnly]
    public readonly unsafe struct NativeTrecsListRead<T>
        where T : unmanaged
    {
        [NativeDisableUnsafePtrRestriction]
        readonly TrecsListHeader* _header;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        readonly AtomicSafetyHandle m_Safety;

        static readonly SharedStatic<int> s_staticSafetyId = SharedStatic<int>.GetOrCreate<
            NativeTrecsListRead<T>
        >();

        internal NativeTrecsListRead(TrecsListHeader* header, AtomicSafetyHandle safety)
        {
            _header = header;
            m_Safety = safety;
            CollectionHelper.SetStaticSafetyId<NativeTrecsListRead<T>>(
                ref m_Safety,
                ref s_staticSafetyId.Data
            );
        }
#else
        internal NativeTrecsListRead(TrecsListHeader* header)
        {
            _header = header;
        }
#endif

        public int Count
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                AtomicSafetyHandle.CheckReadAndThrow(m_Safety);
#endif
                return _header->Count;
            }
        }

        public int Capacity
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                AtomicSafetyHandle.CheckReadAndThrow(m_Safety);
#endif
                return _header->Capacity;
            }
        }

        public ref readonly T this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                AtomicSafetyHandle.CheckReadAndThrow(m_Safety);
#endif
                TrecsDebugAssert.That(
                    (uint)index < (uint)_header->Count,
                    "NativeTrecsListRead index {0} out of range (Count={1})",
                    index,
                    _header->Count
                );
                return ref TrecsListLayout<T>.Data(_header)[index];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public NativeTrecsListReadEnumerator<T> GetEnumerator() =>
            new NativeTrecsListReadEnumerator<T>(this);
    }

    public struct NativeTrecsListReadEnumerator<T>
        where T : unmanaged
    {
        NativeTrecsListRead<T> _view;
        int _index;

        public NativeTrecsListReadEnumerator(NativeTrecsListRead<T> view)
        {
            _view = view;
            _index = -1;
        }

        public ref readonly T Current
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
}
