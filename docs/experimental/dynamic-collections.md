# Dynamic Collections

!!! warning "Experimental"
    The Trecs dynamic-collection family is experimental — API and backing-storage details may change in future 0.x releases.

## Why native collections can't sit directly in a component

Native collection types (`NativeList<T>`, `NativeHashMap<K,V>`, `NativeQueue<T>`, etc.) hold a direct pointer to a memory address. Trecs serializes components as raw memory and expects them to be self-contained — a bare `NativeList` field would write its pointer bytes, not the elements behind them, and snapshots/recordings would silently drop the contents.

```csharp
public partial struct CCollisionPairBuffer : IEntityComponent
{
    // Don't do this — the NativeList pointer won't survive serialization
    public NativeList<CollisionPair> Value;
}
```

For collections with known bounds, [`FixedList<N>`](../advanced/fixed-collections.md) is usually simplest — it stores data inline in the component and needs no manual disposal.

For dynamically-sized data, Trecs provides its own collection types: `TrecsList<T>`, `TrecsArray<T>`, and `TrecsDictionary<TKey, TValue>`. Their backing storage lives on a dedicated heap that Trecs walks during snapshot / record / playback, so the contents round-trip automatically alongside the component bytes — no `NativeUniquePtr` wrapper or custom serializer needed.

## Common patterns

All three collection types share these traits:

- **Handle-sized structs.** The struct on the component is a lightweight handle (4-8 bytes). Backing data lives on the world's native heap.
- **Read/Write wrappers.** Access goes through `Read(...)` / `Write(...)` methods that return typed wrappers, so Unity's job-safety system can track conflicts.
- **Managed and native flavors.** Main-thread wrappers (`TrecsListWrite<T>`, etc.) and Burst-safe wrappers (`NativeTrecsListWrite<T>`, etc.) both grow on overflow, in a job too. The handle never changes. `AddNoResize` (and `TryAddNoResize`, `GetOrAddNoResize` on dictionaries) skip the capacity check for hot loops that sized the collection first.
- **One heap slot per collection.** Header and elements share one block. Capacity fills the slot, so `Alloc` with capacity 0 still holds a few small elements.
- **Manual disposal.** Not auto-freed when an entity is removed. Dispose in an `OnRemoved` observer — see [Pointers — cleanup is manual](pointers.md#cleanup-is-manual-for-entity-owned-pointers).
- **`Write` is `ref this`.** The call site needs a writable reference to the handle — same rule as [the gotcha for `NativeUniquePtr`](../guides/gotchas.md#mutating-a-nativeuniqueptrt-needs-write-access-to-the-owning-component).

## `TrecsList<T>`

A growable list of unmanaged values. 4-byte handle.

```csharp
public partial struct CCollisionPairBuffer : IEntityComponent
{
    public TrecsList<CollisionPair> Value;
}
```

### Allocating

```csharp
var list = TrecsList.Alloc<CollisionPair>(World, initialCapacity: 16);

World.AddEntity<MyTag>()
    .Set(new CCollisionPairBuffer { Value = list });
```

### Reading and writing

```csharp
// Main thread
ref readonly var entry = ref list.Read(World)[0];

var write = list.Write(World);
write.Add(new CollisionPair(a, b));
write[0] = updatedEntry;
write.RemoveAtSwapBack(2);
write.Clear();
```

Inside a Burst job:

```csharp
[ForEachEntity(typeof(MyTag))]
[WrapAsJob]
static void Execute(ref CCollisionPairBuffer buf, in NativeWorldAccessor world)
{
    var write = buf.Value.Write(world);
    write.Add(/* ... */);
}
```

### Capacity

`Add` doubles the capacity when full, on the main thread and in Burst jobs. A job grows into a block outside the heap's pages; the heap moves it back into its pages at the next phase boundary, in handle order, so snapshots and lockstep peers stay identical. `list.EnsureCapacity(World, minCapacity)` sizes up front; `AddNoResize` then skips the capacity check.

## `TrecsArray<T>`

A fixed-size array of unmanaged values. 8-byte handle (handle + inline length). The size is locked at allocation and never grows — use this when the element count is known up front but too large to inline via `FixedArray<N>`.

```csharp
public partial struct CWaypoints : IEntityComponent
{
    public TrecsArray<float3> Value;
}
```

### Allocating

```csharp
var arr = TrecsArray.Alloc<float3>(World, length: 64);

World.AddEntity<MyTag>()
    .Set(new CWaypoints { Value = arr });
```

### Reading and writing

```csharp
// Main thread
ref readonly var point = ref arr.Read(World)[0];

var write = arr.Write(World);
write[0] = new float3(1, 2, 3);
```

`TrecsArrayRead<T>` and `TrecsArrayWrite<T>` are Burst-safe — the same wrappers work on the main thread and in jobs. There is no `Add` or `Remove`; only indexed access.

## `TrecsDictionary<TKey, TValue>`

A growable hash dictionary. 4-byte handle. Keys must implement `IEquatable<TKey>`.

```csharp
public partial struct CInventory : IEntityComponent
{
    public TrecsDictionary<int, int> Value;  // item ID → count
}
```

### Allocating

```csharp
var dict = TrecsDictionary.Alloc<int, int>(World, initialCapacity: 8);

World.AddEntity<MyTag>()
    .Set(new CInventory { Value = dict });
```

### Reading and writing

```csharp
// Main thread
var read = dict.Read(World);
if (read.TryGetValue(itemId, out int count)) { /* ... */ }

var write = dict.Write(World);
write.Add(itemId, 5);
write[itemId] = 10;
write.Remove(itemId);
write.Clear();
```

Other write operations: `TryAdd`, `Set` (update existing, throws if missing), `GetOrAdd` (get-or-create returning `ref TValue`), `GetValueByRef`.

Inside a Burst job, `NativeTrecsDictionaryWrite` provides the same API and grows the same way as a list.

### Add vs TryAdd vs Set vs indexer

```csharp
var write = dict.Write(World);

write.Add(42, 100);                   // insert only; debug-assert on duplicate
bool added = write.TryAdd(42, 200);   // insert only; silent no-op on duplicate
write.Set(42, 300);                   // update only; throws if key missing
write[99] = 500;                      // add-or-update (most permissive)
```

### Index-based access

For advanced patterns — e.g., correlating a dictionary entry with data in a parallel array — use the dense-array index API:

```csharp
var write = dict.Write(World);
if (write.TryGetIndex(key, out int idx))
{
    ref var val = ref write.GetValueAtIndex(idx);
    val.Score += 10;
}
```

## Disposing

All three types must be manually disposed. Dispose in an `OnRemoved` observer:

```csharp
[ForEachEntity]
void OnEntityRemoved(in CCollisionPairBuffer buf, in CInventory inv)
{
    buf.Value.Dispose(World);
    inv.Value.Dispose(World);
}
```

Forgetting to dispose leaks the backing storage; Trecs reports leaks at world shutdown in debug/editor builds.

## When to reach for each option

| Use | When |
|---|---|
| [`FixedList<N>`](../advanced/fixed-collections.md) / [`FixedArray<N>`](../advanced/fixed-collections.md) | Known small upper bound. Inline, no disposal. |
| `TrecsArray<T>` | Fixed count known at allocation, too large to inline. |
| `TrecsList<T>` | Variable per-entity count. |
| `TrecsDictionary<TKey, TValue>` | Key-value lookups. |
| `UniquePtr<List<T>>` | Managed element types (classes, strings). Main-thread only. Needs a registered `ISerializer<T>` — see [Serialization](serialization.md). |
