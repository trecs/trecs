using System;
using System.Runtime.CompilerServices;
using Trecs.Internal;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Trecs
{
    /// <summary>
    /// Burst-safe read-only view over a <see cref="TrecsDictionary{TKey, TValue}"/>. Caches
    /// the block pointer: nothing can grow the dictionary while a reader holds it.
    /// </summary>
    [NativeContainer]
    [NativeContainerIsReadOnly]
    public readonly unsafe struct NativeTrecsDictionaryRead<TKey, TValue>
        where TKey : unmanaged, IEquatable<TKey>
        where TValue : unmanaged
    {
        [NativeDisableUnsafePtrRestriction]
        readonly TrecsDictionaryHeader* _header;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        readonly AtomicSafetyHandle m_Safety;

        static readonly SharedStatic<int> s_staticSafetyId = SharedStatic<int>.GetOrCreate<
            NativeTrecsDictionaryRead<TKey, TValue>
        >();

        internal NativeTrecsDictionaryRead(TrecsDictionaryHeader* header, AtomicSafetyHandle safety)
        {
            _header = header;
            m_Safety = safety;
            CollectionHelper.SetStaticSafetyId<NativeTrecsDictionaryRead<TKey, TValue>>(
                ref m_Safety,
                ref s_staticSafetyId.Data
            );
        }
#else
        internal NativeTrecsDictionaryRead(TrecsDictionaryHeader* header)
        {
            _header = header;
        }
#endif

        TrecsDictionaryHeader* Header
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                AtomicSafetyHandle.CheckReadAndThrow(m_Safety);
#endif
                return _header;
            }
        }

        public int Count
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Header->Count;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool ContainsKey(TKey key) =>
            TrecsDictionaryLayout<TKey, TValue>.TryGetIndex(Header, key, out _);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryGetValue(TKey key, out TValue value)
        {
            var header = Header;
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
                var header = Header;
                TrecsAssert.That(
                    TrecsDictionaryLayout<TKey, TValue>.TryGetIndex(header, key, out var index),
                    "Key not found in TrecsDictionary"
                );
                return TrecsDictionaryLayout<TKey, TValue>.Values(header)[index];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryGetIndex(TKey key, out int index) =>
            TrecsDictionaryLayout<TKey, TValue>.TryGetIndex(Header, key, out index);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref readonly TValue GetValueAtIndex(int index)
        {
            var header = Header;
            TrecsDebugAssert.That(
                (uint)index < (uint)header->Count,
                "NativeTrecsDictionaryRead.GetValueAtIndex index {0} out of range (Count={1})",
                index,
                header->Count
            );
            return ref TrecsDictionaryLayout<TKey, TValue>.Values(header)[index];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TKey GetKeyAtIndex(int index)
        {
            var header = Header;
            TrecsDebugAssert.That(
                (uint)index < (uint)header->Count,
                "NativeTrecsDictionaryRead.GetKeyAtIndex index {0} out of range (Count={1})",
                index,
                header->Count
            );
            return TrecsDictionaryLayout<TKey, TValue>.Nodes(header)[index].Key;
        }

        public TrecsDictionaryKeys<TKey> Keys
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                var header = Header;
                return new TrecsDictionaryKeys<TKey>(
                    TrecsDictionaryLayout<TKey, TValue>.Nodes(header),
                    header->Count
                );
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TrecsDictionaryReadEnumerator<TKey, TValue> GetEnumerator()
        {
            var header = Header;
            return new TrecsDictionaryReadEnumerator<TKey, TValue>(
                TrecsDictionaryLayout<TKey, TValue>.Nodes(header),
                TrecsDictionaryLayout<TKey, TValue>.Values(header),
                header->Count
            );
        }
    }
}
