using System.Runtime.CompilerServices;
using Trecs.Internal;
using Unity.Collections.LowLevel.Unsafe;

namespace Trecs
{
    /// <summary>
    /// Main-thread read-only view over a <see cref="TrecsList{T}"/>. In a job use
    /// <see cref="NativeTrecsListRead{T}"/>.
    ///
    /// <para>Reaches the block through the allocation's side-table entry on every access, so
    /// it stays valid when the list grows through another view, and asserts the allocation
    /// is still live (shipping builds too). A <c>ref struct</c>: it cannot outlive the
    /// method that opened it.</para>
    /// </summary>
    public readonly unsafe ref struct TrecsListRead<T>
        where T : unmanaged
    {
        readonly NativeHeapEntry* _slot;
        readonly PtrHandle _handle;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        readonly AtomicSafetyHandle m_Safety;

        internal TrecsListRead(NativeHeapEntry* slot, PtrHandle handle, AtomicSafetyHandle safety)
        {
            _slot = slot;
            _handle = handle;
            m_Safety = safety;
        }
#else
        internal TrecsListRead(NativeHeapEntry* slot, PtrHandle handle)
        {
            _slot = slot;
            _handle = handle;
        }
#endif

        TrecsListHeader* Header
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

        public int Count
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Header->Count;
        }

        public int Capacity
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Header->Capacity;
        }

        public ref readonly T this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                var header = Header;
                TrecsDebugAssert.That(
                    (uint)index < (uint)header->Count,
                    "TrecsListRead index {0} out of range (Count={1})",
                    index,
                    header->Count
                );
                return ref TrecsListLayout<T>.Data(header)[index];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TrecsListReadEnumerator<T> GetEnumerator() => new TrecsListReadEnumerator<T>(this);
    }

    public ref struct TrecsListReadEnumerator<T>
        where T : unmanaged
    {
        TrecsListRead<T> _view;
        int _index;

        public TrecsListReadEnumerator(TrecsListRead<T> view)
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
