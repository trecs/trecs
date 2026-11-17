using System;
using System.Diagnostics;
using System.IO;
using NUnit.Framework;
using Trecs.Internal;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Trecs.Tests
{
    /// <summary>
    /// TrecsList against UnsafeList, one list per entity: held inline (a component field) and
    /// through a pointer (NativeList's layout). Logs one warning per measure.
    /// </summary>
    [TestFixture]
    [Explicit]
    [Category("Benchmark")]
    public unsafe class HeapContainerBenchmarks
    {
        const int Runs = 21;
        const int Batch = 64;

        [TestCase(10_000, 8, true)]
        [TestCase(100_000, 8, true)]
        [TestCase(100_000, 64, true)]
        [TestCase(100_000, 8, false)]
        [TestCase(100_000, 64, false)]
        public void TrecsListAgainstUnsafeList(int entityCount, int length, bool safetyChecks)
        {
            var safetyBefore = BurstCompiler.Options.EnableBurstSafetyChecks;
            BurstCompiler.Options.EnableBurstSafetyChecks = safetyChecks;
            using var heap = new NativeHeap(TrecsLog.Default);
            var trecs = new NativeArray<TrecsList<int>>(entityCount, Allocator.Persistent);
            var inline = new NativeArray<UnsafeList<int>>(entityCount, Allocator.Persistent);
            var pointers = new NativeArray<IntPtr>(entityCount, Allocator.Persistent);
            var sums = new NativeArray<int>(entityCount, Allocator.Persistent);
            try
            {
                for (int i = 0; i < entityCount; i++)
                {
                    trecs[i] = TrecsList.Alloc<int>(heap, length);
                    inline[i] = new UnsafeList<int>(length, Allocator.Persistent);
                    var list = (UnsafeList<int>*)
                        UnsafeUtility.MallocTracked(
                            sizeof(UnsafeList<int>),
                            UnsafeUtility.AlignOf<UnsafeList<int>>(),
                            Allocator.Persistent,
                            0
                        );
                    *list = new UnsafeList<int>(length, Allocator.Persistent);
                    pointers[i] = (IntPtr)list;
                }
                var resolver = heap.Resolver;
                var label = $"{entityCount} lists x {length} ints, safety {(safetyChecks ? "on" : "off")}";

                var fillTrecs = new RefillTrecs
                {
                    Lists = trecs,
                    Resolver = resolver,
                    Length = length,
                };
                var fillTrecsNoResize = fillTrecs;
                fillTrecsNoResize.NoResize = true;
                var fillInline = new RefillInline { Lists = inline, Length = length };
                var fillPointer = new RefillPointer { Lists = pointers, Length = length };
                Report(label, "refill Add", Time(fillTrecs), Time(fillInline), Time(fillPointer));
                Report(
                    label,
                    "refill AddNoResize",
                    Time(fillTrecsNoResize),
                    Time(new RefillInline { Lists = inline, Length = length, NoResize = true }),
                    Time(new RefillPointer { Lists = pointers, Length = length, NoResize = true })
                );

                Report(
                    label,
                    "sum (read view)",
                    Time(new SumTrecsRead { Lists = trecs, Resolver = resolver, Sums = sums }),
                    Time(new SumInline { Lists = inline, Sums = sums }),
                    Time(new SumPointer { Lists = pointers, Sums = sums })
                );
                Report(
                    label,
                    "increment (write view)",
                    Time(new IncrementTrecs { Lists = trecs, Resolver = resolver }),
                    Time(new IncrementInline { Lists = inline }),
                    Time(new IncrementPointer { Lists = pointers })
                );

                // Growth from the smallest block: Trecs grows into detached blocks, then
                // adopts them on the main thread.
                var growTrecs = new double[Runs];
                var adoptTrecs = new double[Runs];
                var growInline = new double[Runs];
                var growPointer = new double[Runs];
                for (int run = 0; run < Runs; run++)
                {
                    ShrinkAll(heap, trecs, inline, pointers);
                    growTrecs[run] = TimeOnce(fillTrecs);
                    var watch = Stopwatch.StartNew();
                    heap.AdoptDetached();
                    adoptTrecs[run] = watch.Elapsed.TotalMilliseconds;
                    growInline[run] = TimeOnce(fillInline);
                    growPointer[run] = TimeOnce(fillPointer);
                }
                Report(label, "grow from empty", growTrecs, growInline, growPointer);
                Debug.LogWarning($"[bench] {label} | grow from empty | trecs adopt {Median(adoptTrecs):F3} ms");

                ReportMemory(label, heap, inline);
                ReportSerialization(label, heap, trecs, inline);
            }
            finally
            {
                for (int i = 0; i < entityCount; i++)
                {
                    trecs[i].Dispose(heap);
                    inline[i].Dispose();
                    var list = (UnsafeList<int>*)pointers[i];
                    list->Dispose();
                    UnsafeUtility.FreeTracked(list, Allocator.Persistent);
                }
                trecs.Dispose();
                inline.Dispose();
                pointers.Dispose();
                sums.Dispose();
                BurstCompiler.Options.EnableBurstSafetyChecks = safetyBefore;
            }
        }

        static void ShrinkAll(
            NativeHeap heap,
            NativeArray<TrecsList<int>> trecs,
            NativeArray<UnsafeList<int>> inline,
            NativeArray<IntPtr> pointers
        )
        {
            for (int i = 0; i < trecs.Length; i++)
            {
                trecs[i].Dispose(heap);
                trecs[i] = TrecsList.Alloc<int>(heap);
                inline[i].Dispose();
                inline[i] = new UnsafeList<int>(0, Allocator.Persistent);
                var list = (UnsafeList<int>*)pointers[i];
                list->Dispose();
                *list = new UnsafeList<int>(0, Allocator.Persistent);
            }
        }

        static void ReportSerialization(
            string label,
            NativeHeap heap,
            NativeArray<TrecsList<int>> trecs,
            NativeArray<UnsafeList<int>> inline
        )
        {
            var registry = new SerializerRegistry();
            var trecsWrite = new double[Runs];
            var inlineWrite = new double[Runs];
            byte[] trecsBytes = null;
            byte[] inlineBytes = null;
            for (int run = 0; run < Runs; run++)
            {
                var writer = new BinarySerializationWriter(registry);
                writer.Start(version: 1, includeTypeChecks: false);
                var watch = Stopwatch.StartNew();
                heap.Serialize(writer);
                trecsWrite[run] = watch.Elapsed.TotalMilliseconds;
                trecsBytes = Complete(writer);

                writer = new BinarySerializationWriter(registry);
                writer.Start(version: 1, includeTypeChecks: false);
                watch.Restart();
                WriteInline(writer, inline);
                inlineWrite[run] = watch.Elapsed.TotalMilliseconds;
                inlineBytes = Complete(writer);
            }
            Report(label, "serialize", trecsWrite, inlineWrite, null);

            // Restore over live state, as a rollback does.
            var trecsRead = new double[Runs];
            var inlineRead = new double[Runs];
            var inlineReuse = new double[Runs];
            for (int run = 0; run < Runs; run++)
            {
                var reader = new BinarySerializationReader(registry);
                reader.Start(trecsBytes);
                var watch = Stopwatch.StartNew();
                heap.Deserialize(reader);
                trecsRead[run] = watch.Elapsed.TotalMilliseconds;

                reader = new BinarySerializationReader(registry);
                reader.Start(inlineBytes);
                watch.Restart();
                ReadInline(reader, inline, reuse: false);
                inlineRead[run] = watch.Elapsed.TotalMilliseconds;

                reader = new BinarySerializationReader(registry);
                reader.Start(inlineBytes);
                watch.Restart();
                ReadInline(reader, inline, reuse: true);
                inlineReuse[run] = watch.Elapsed.TotalMilliseconds;
            }
            Report(label, "deserialize over live state", trecsRead, inlineRead, null);
            Debug.LogWarning(
                $"[bench] {label} | deserialize, unsafe reusing buffers | {Median(inlineReuse):F3} ms"
            );
            Debug.LogWarning(
                $"[bench] {label} | snapshot bytes | trecs {trecsBytes.Length} unsafe {inlineBytes.Length}"
            );
        }

        static void WriteInline(BinarySerializationWriter writer, NativeArray<UnsafeList<int>> lists)
        {
            for (int i = 0; i < lists.Length; i++)
            {
                var list = lists[i];
                var count = list.Length;
                writer.BlitWriteRawBytes("Count", &count, sizeof(int));
                writer.BlitWriteRawBytes("Data", list.Ptr, count * sizeof(int));
            }
        }

        static void ReadInline(
            BinarySerializationReader reader,
            NativeArray<UnsafeList<int>> lists,
            bool reuse
        )
        {
            for (int i = 0; i < lists.Length; i++)
            {
                int count;
                reader.BlitReadRawBytes("Count", &count, sizeof(int));
                var list = lists[i];
                if (!reuse || list.Capacity < count)
                {
                    list.Dispose();
                    list = new UnsafeList<int>(count, Allocator.Persistent);
                }
                list.Length = count;
                reader.BlitReadRawBytes("Data", list.Ptr, count * sizeof(int));
                lists[i] = list;
            }
        }

        static void ReportMemory(string label, NativeHeap heap, NativeArray<UnsafeList<int>> inline)
        {
            var stats = heap.GetStatistics();
            var sideTable =
                (long)((stats.SideTableLength + NativeHeapResolver.ChunkSize - 1) / NativeHeapResolver.ChunkSize)
                * NativeHeapResolver.ChunkByteSize;
            var perAllocation = stats.LiveAllocations * (16L + 4 + 4); // dense payload, slot index, sparse map
            var directory = NativeHeapResolver.MaxChunkCount * (long)IntPtr.Size;
            long unsafeBytes = 0;
            for (int i = 0; i < inline.Length; i++)
            {
                unsafeBytes += (long)inline[i].Capacity * sizeof(int);
            }
            Debug.LogWarning(
                $"[bench] {label} | memory | trecs pages {stats.TotalReservedBytes} + side table {sideTable}"
                    + $" + bookkeeping {perAllocation} + directory {directory}"
                    + $" | unsafe buffers {unsafeBytes} + inline structs {inline.Length * (long)sizeof(UnsafeList<int>)}"
                    + " + malloc headers (not measured)"
            );
        }

        static byte[] Complete(BinarySerializationWriter writer)
        {
            using var stream = new MemoryStream();
            writer.Complete(stream);
            return stream.ToArray();
        }

        static double[] Time<TJob>(TJob job)
            where TJob : struct, IJobParallelFor
        {
            TimeOnce(job);
            var times = new double[Runs];
            for (int run = 0; run < Runs; run++)
            {
                times[run] = TimeOnce(job);
            }
            return times;
        }

        static double TimeOnce<TJob>(TJob job)
            where TJob : struct, IJobParallelFor
        {
            var length = JobLength(job);
            var watch = Stopwatch.StartNew();
            job.Schedule(length, Batch).Complete();
            return watch.Elapsed.TotalMilliseconds;
        }

        static int JobLength<TJob>(TJob job) =>
            job switch
            {
                RefillTrecs j => j.Lists.Length,
                RefillInline j => j.Lists.Length,
                RefillPointer j => j.Lists.Length,
                SumTrecsRead j => j.Lists.Length,
                SumInline j => j.Lists.Length,
                SumPointer j => j.Lists.Length,
                IncrementTrecs j => j.Lists.Length,
                IncrementInline j => j.Lists.Length,
                IncrementPointer j => j.Lists.Length,
                _ => throw new ArgumentException(typeof(TJob).Name),
            };

        static double Median(double[] values)
        {
            var sorted = (double[])values.Clone();
            Array.Sort(sorted);
            return sorted[sorted.Length / 2];
        }

        static void Report(string label, string measure, double[] trecs, double[] inline, double[] pointer)
        {
            var line = $"[bench] {label} | {measure} | trecs {Median(trecs):F3} ms | unsafe inline {Median(inline):F3} ms";
            if (pointer != null)
            {
                line += $" | unsafe pointer {Median(pointer):F3} ms";
            }
            Debug.LogWarning(line);
        }

        [BurstCompile(CompileSynchronously = true)]
        struct RefillTrecs : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<TrecsList<int>> Lists;
            public NativeHeapResolver Resolver;
            public int Length;
            public bool NoResize;

            public void Execute(int index)
            {
                var list = Lists[index];
                var write = list.Write(in Resolver);
                write.Clear();
                if (NoResize)
                {
                    for (int i = 0; i < Length; i++)
                        write.AddNoResize(index + i);
                }
                else
                {
                    for (int i = 0; i < Length; i++)
                        write.Add(index + i);
                }
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct RefillInline : IJobParallelFor
        {
            [NativeDisableParallelForRestriction]
            public NativeArray<UnsafeList<int>> Lists;
            public int Length;
            public bool NoResize;

            public void Execute(int index)
            {
                ref var list = ref UnsafeUtility.ArrayElementAsRef<UnsafeList<int>>(
                    Lists.GetUnsafePtr(),
                    index
                );
                list.Clear();
                if (NoResize)
                {
                    for (int i = 0; i < Length; i++)
                        list.AddNoResize(index + i);
                }
                else
                {
                    for (int i = 0; i < Length; i++)
                        list.Add(index + i);
                }
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct RefillPointer : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<IntPtr> Lists;
            public int Length;
            public bool NoResize;

            public void Execute(int index)
            {
                var list = (UnsafeList<int>*)Lists[index];
                list->Clear();
                if (NoResize)
                {
                    for (int i = 0; i < Length; i++)
                        list->AddNoResize(index + i);
                }
                else
                {
                    for (int i = 0; i < Length; i++)
                        list->Add(index + i);
                }
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct SumTrecsRead : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<TrecsList<int>> Lists;
            public NativeHeapResolver Resolver;
            public NativeArray<int> Sums;

            public void Execute(int index)
            {
                var read = Lists[index].Read(in Resolver);
                int sum = 0;
                for (int i = 0; i < read.Count; i++)
                    sum += read[i];
                Sums[index] = sum;
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct SumInline : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<UnsafeList<int>> Lists;
            public NativeArray<int> Sums;

            public void Execute(int index)
            {
                var list = Lists[index];
                int sum = 0;
                for (int i = 0; i < list.Length; i++)
                    sum += list[i];
                Sums[index] = sum;
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct SumPointer : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<IntPtr> Lists;
            public NativeArray<int> Sums;

            public void Execute(int index)
            {
                var list = (UnsafeList<int>*)Lists[index];
                int sum = 0;
                for (int i = 0; i < list->Length; i++)
                    sum += (*list)[i];
                Sums[index] = sum;
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct IncrementTrecs : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<TrecsList<int>> Lists;
            public NativeHeapResolver Resolver;

            public void Execute(int index)
            {
                var list = Lists[index];
                var write = list.Write(in Resolver);
                for (int i = 0; i < write.Count; i++)
                    write[i] += 1;
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct IncrementInline : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<UnsafeList<int>> Lists;

            public void Execute(int index)
            {
                var list = Lists[index];
                for (int i = 0; i < list.Length; i++)
                    list[i] += 1;
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct IncrementPointer : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<IntPtr> Lists;

            public void Execute(int index)
            {
                var list = (UnsafeList<int>*)Lists[index];
                for (int i = 0; i < list->Length; i++)
                    (*list)[i] += 1;
            }
        }
    }
}
