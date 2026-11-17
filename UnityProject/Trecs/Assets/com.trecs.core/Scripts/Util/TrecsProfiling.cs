using System;
using Unity.Profiling;
using UnityEngine.Profiling;

namespace Trecs.Internal
{
    public static class TrecsProfiling
    {
        // Looks the marker up by name each call, ~100 ns more than a TrecsProfilerMarker.
        // Per-entity, per-allocation, per-system scopes: TrecsProfilerMarker.
        public static Scope Start(string name)
        {
            Profiler.BeginSample(name);
            return default;
        }

        public readonly struct Scope : IDisposable
        {
            public void Dispose() => Profiler.EndSample();
        }
    }

    // Name must be fixed per scope: each distinct name creates a marker for the process lifetime.
    public readonly struct TrecsProfilerMarker
    {
#if ENABLE_PROFILER
        readonly ProfilerMarker _marker;

        public TrecsProfilerMarker(string name) => _marker = new ProfilerMarker(name);

        public ProfilerMarker.AutoScope Auto() => _marker.Auto();
#else
        public TrecsProfilerMarker(string name) { }

        public TrecsProfiling.Scope Auto() => default;
#endif
    }
}
