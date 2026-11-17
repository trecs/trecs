using System.IO;
using NUnit.Framework;
using Trecs.Internal;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using NAssert = NUnit.Framework.Assert;

namespace Trecs.Tests
{
    [TestFixture]
    public class HeapDetachedGrowthTests
    {
        [BurstCompile(CompileSynchronously = true)]
        struct AppendJob : IJob
        {
            public NativeTrecsListWrite<int> List;
            public int Count;

            public void Execute()
            {
                for (int i = 0; i < Count; i++)
                {
                    List.Add(i);
                }
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct AppendToEachJob : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<TrecsList<int>> Lists;
            public NativeHeapResolver Resolver;

            public void Execute(int index)
            {
                var list = Lists[index];
                var write = list.Write(in Resolver);
                for (int i = 0; i < 5 + index * 3; i++)
                {
                    write.Add(index * 1000 + i);
                }
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct InsertJob : IJob
        {
            public NativeTrecsDictionaryWrite<int, int> Dict;
            public int Count;

            public void Execute()
            {
                for (int i = 0; i < Count; i++)
                {
                    Dict.Add(i * 7, i);
                }
                Dict.Remove(7);
            }
        }

        [Test]
        public void NativeListAdd_PastCapacityInJob_GrowsWithSameHandle()
        {
            using var heap = new NativeHeap(TrecsLog.Default);
            var list = TrecsList.Alloc<int>(heap);

            new AppendJob { List = list.Write(heap.Resolver), Count = 1000 }
                .Schedule()
                .Complete();

            NAssert.AreEqual(1, heap.NumLiveAllocations);
            AssertSequence(list, heap, 1000);
            heap.AdoptDetached();
            AssertSequence(list, heap, 1000);

            list.Dispose(heap);
        }

        [Test]
        public void NativeDictionaryAdd_PastCapacityInJob_Rehashes()
        {
            using var heap = new NativeHeap(TrecsLog.Default);
            var dict = TrecsDictionary.Alloc<int, int>(heap);

            new InsertJob { Dict = dict.Write(heap.Resolver), Count = 500 }
                .Schedule()
                .Complete();
            heap.AdoptDetached();

            var read = dict.Read(heap);
            NAssert.AreEqual(499, read.Count);
            NAssert.IsFalse(read.ContainsKey(7));
            for (int i = 0; i < 500; i++)
            {
                if (i != 1)
                {
                    NAssert.AreEqual(i, read[i * 7]);
                }
            }

            dict.Dispose(heap);
        }

        [Test]
        public void OneList_GrownByTwoJobsAndMainThread_ThenNextPhase_KeepsElements()
        {
            using var heap = new NativeHeap(TrecsLog.Default);
            var list = TrecsList.Alloc<int>(heap);

            // Phase 1: a second job grows the block the first job detached; then the main thread.
            new AppendJob { List = list.Write(heap.Resolver), Count = 100 }
                .Schedule()
                .Complete();
            new AppendNextJob { List = list.Write(heap.Resolver), Count = 300 }
                .Schedule()
                .Complete();
            var write = list.Write(heap);
            for (int i = 400; i < 1000; i++)
            {
                write.Add(i);
            }
            heap.AdoptDetached();
            AssertSequence(list, heap, 1000);

            // Phase 2: grows again from its page slot.
            new AppendNextJob { List = list.Write(heap.Resolver), Count = 3000 }
                .Schedule()
                .Complete();
            heap.AdoptDetached();
            AssertSequence(list, heap, 4000);
            NAssert.AreEqual(1, heap.NumLiveAllocations);

            list.Dispose(heap);
        }

        [BurstCompile(CompileSynchronously = true)]
        struct AppendNextJob : IJob
        {
            public NativeTrecsListWrite<int> List;
            public int Count;

            public void Execute()
            {
                var start = List.Count;
                for (int i = 0; i < Count; i++)
                {
                    List.Add(start + i);
                }
            }
        }

        [Test]
        public void NativeInsert_PastCapacityInJob_GrowsAndShifts()
        {
            using var heap = new NativeHeap(TrecsLog.Default);
            var list = TrecsList.Alloc<int>(heap);
            new InsertFrontJob { List = list.Write(heap.Resolver), Count = 200 }
                .Schedule()
                .Complete();
            heap.AdoptDetached();

            var read = list.Read(heap);
            NAssert.AreEqual(200, read.Count);
            for (int i = 0; i < 200; i++)
            {
                NAssert.AreEqual(199 - i, read[i]);
            }
            list.Dispose(heap);
        }

        [BurstCompile(CompileSynchronously = true)]
        struct InsertFrontJob : IJob
        {
            public NativeTrecsListWrite<int> List;
            public int Count;

            public void Execute()
            {
                for (int i = 0; i < Count; i++)
                {
                    List.Insert(0, i);
                }
            }
        }

        [Test]
        public void JobGrowth_SerializedHeapIsSameForAnyJobOrder()
        {
            var forward = GrowInJobsThenSerialize(reverseOrder: false);
            var reverse = GrowInJobsThenSerialize(reverseOrder: true);
            CollectionAssert.AreEqual(forward, reverse);
        }

        [Test]
        public void DetachedList_DisposedBeforeAdopt_IsFreed()
        {
            using var heap = new NativeHeap(TrecsLog.Default);
            var list = TrecsList.Alloc<int>(heap);
            new AppendJob { List = list.Write(heap.Resolver), Count = 100 }
                .Schedule()
                .Complete();

            list.Dispose(heap);
            heap.AdoptDetached();

            NAssert.AreEqual(0, heap.NumLiveAllocations);
        }

        [Test]
        public void MainThreadGrow_OfDetachedList_KeepsElements()
        {
            using var heap = new NativeHeap(TrecsLog.Default);
            var list = TrecsList.Alloc<int>(heap);
            new AppendJob { List = list.Write(heap.Resolver), Count = 100 }
                .Schedule()
                .Complete();

            list.EnsureCapacity(heap, 5000);
            heap.AdoptDetached();

            AssertSequence(list, heap, 100);
            NAssert.GreaterOrEqual(list.Read(heap).Capacity, 5000);
            list.Dispose(heap);
        }

        static byte[] GrowInJobsThenSerialize(bool reverseOrder)
        {
            using var heap = new NativeHeap(TrecsLog.Default);
            const int listCount = 64;
            var allocated = new TrecsList<int>[listCount];
            for (int i = 0; i < listCount; i++)
            {
                allocated[i] = TrecsList.Alloc<int>(heap);
            }
            using var lists = new NativeArray<TrecsList<int>>(allocated, Allocator.TempJob);

            if (reverseOrder)
            {
                // Same growth, other threads and order: one job per list, last list first.
                for (int i = listCount - 1; i >= 0; i--)
                {
                    using var one = new NativeArray<TrecsList<int>>(
                        new[] { lists[i] },
                        Allocator.TempJob
                    );
                    new AppendToEachOffsetJob
                    {
                        Lists = one,
                        Resolver = heap.Resolver,
                        Offset = i,
                    }
                        .Schedule(1, 1)
                        .Complete();
                }
            }
            else
            {
                new AppendToEachJob { Lists = lists, Resolver = heap.Resolver }
                    .Schedule(listCount, 1)
                    .Complete();
            }
            heap.AdoptDetached();

            var writer = new BinarySerializationWriter(new SerializerRegistry());
            writer.Start(version: 1, includeTypeChecks: false);
            heap.Serialize(writer);
            using var stream = new MemoryStream();
            writer.Complete(stream);

            for (int i = 0; i < listCount; i++)
            {
                lists[i].Dispose(heap);
            }
            return stream.ToArray();
        }

        // AppendToEachJob for one list at index Offset of the full set.
        [BurstCompile(CompileSynchronously = true)]
        struct AppendToEachOffsetJob : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<TrecsList<int>> Lists;
            public NativeHeapResolver Resolver;
            public int Offset;

            public void Execute(int index)
            {
                var list = Lists[index];
                var write = list.Write(in Resolver);
                var at = index + Offset;
                for (int i = 0; i < 5 + at * 3; i++)
                {
                    write.Add(at * 1000 + i);
                }
            }
        }

        static void AssertSequence(TrecsList<int> list, NativeHeap heap, int count)
        {
            var read = list.Read(heap);
            NAssert.AreEqual(count, read.Count);
            for (int i = 0; i < count; i++)
            {
                NAssert.AreEqual(i, read[i]);
            }
        }
    }
}
