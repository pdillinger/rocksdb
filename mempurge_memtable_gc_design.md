# Mempurge as Memtable GC

## Background

The current mempurge implementation piggy-backs on flush workflows: it reads a
sealed immutable memtable, builds a purged replacement memtable, and keeps that
replacement in the immutable memtable machinery. That creates correctness and
lifecycle pressure in places that are fundamentally about flushing to storage:
immutable-list ordering, flush picking, manual flush waiting, and L0
file-number ordering.

This sketch explores a different shape: treat mempurge as an internal memtable
garbage collection pass. The logical memtable remains the same flush unit, while
its internal skiplist representation can be rewritten into a newer subversion.

## Goals

- Preserve normal `SuperVersion` semantics.
- Avoid putting mempurge output back into the immutable memtable list.
- Avoid multi-structure read merging on the common read path.
- Keep logical memtable identity, WAL dependencies, and flush ordering attached
  to the original memtable.
- Allow old `SuperVersion`s to continue reading an older memtable subversion
  while a newer subversion is being built.

## Non-Goals

- This is not a near-term patch plan.
- This does not attempt to support multiple simultaneous GC passes on one
  memtable.
- This does not propose changing ordinary skiplist memtables unless the
  memtable is using a GC-capable representation.

## Core Idea

A memtable can have subversions. A new subversion gets a new owning `MemTable`
view with a copied skiplist head, but initially shares most of the old skiplist
nodes. GC migrates the graph left-to-right into new memory, using tagged
skiplist links to coordinate with writes.

Old `SuperVersion`s pin the old subversion. New `SuperVersion`s pin the new
subversion.

The desired high-level invariant is:

```text
A SuperVersion pins a specific memtable subversion, and reads through that
subversion see a complete sorted view up to that subversion's visibility cap.
```

## Tagged Links

The skiplist `next` pointer becomes an atomic tagged link rather than a raw
`Node*`.

Conceptually:

```cpp
struct TaggedLink {
  Node* ptr;
  bool frozen;
  uint1_t generation;  // even/odd
};
```

Reads mask off the tag bits and follow `ptr`. Semantically, reads ignore the
tags. Mechanically, the GC-capable skiplist rep pays a small per-link decode
cost.

Writers may CAS through a link if and only if the link is not frozen. On a
successful insert, the writer publishes links tagged with the active generation.
If CAS observes a frozen link or any unexpected link value, the writer re-seeks
from the active subversion head.

GC freezes old-generation links before migrating or bypassing them. If a writer
wins first, GC sees the inserted node and either migrates it or recognizes it as
already active-generation data.

## Writer/GC Race

If GC wins first:

```text
writer seeks:    A -> B
GC freezes:      A -> frozen(B)
writer CAS:      A -> W fails
writer action:   re-seek from active subversion head
```

If the writer wins first:

```text
writer CAS:      A -> W -> B
GC freeze A->B:  fails
GC action:       re-seek, sees W, and migrates or keeps it
```

This makes it safe for writers to insert into shared old nodes as long as they
never CAS through frozen links. The frozen bit is the concurrency-control
primitive. Pointer ranges are useful for reclamation, but should not be the
primary correctness mechanism.

## Visibility Cap

Each memtable subversion has a maximum sequence number it is allowed to expose:

```cpp
SequenceNumber visible_seq_cap;  // kMaxSequenceNumber for the current view
```

For a `SuperVersion` referencing a capped memtable subversion, reads from that
subversion use:

```cpp
effective_snapshot = std::min(read_snapshot, visible_seq_cap);
```

This handles the case where an old subversion can physically traverse nodes
inserted after the subversion was published. Those nodes may be reachable, but
sequence filtering hides them.

The current writable subversion has an unbounded cap.

## GC Algorithm Sketch

1. Serialize subversion creation at the same kind of point used for
   memtable/SuperVersion publication.
2. Create a new owning memtable/subversion with a copied skiplist head.
3. Set the old subversion's `visible_seq_cap` to the current published sequence
   boundary.
4. Publish the new subversion as the current writable view, with an unbounded
   visibility cap.
5. Walk level 0 left-to-right while maintaining cursors at all skiplist heights.
6. Infer each node's height from the simultaneous per-level iteration.
7. For each node:
   - If it is live and old-generation, copy it into a new arena and splice it
     into new-generation links.
   - If it is already active-generation, avoid copying it.
   - If it is garbage, freeze and bypass it in the new subversion.
8. If GC encounters a frozen/conflicting link, re-seek from the active
   subversion head.
9. Reclaim old arenas only after old `SuperVersion`s/subversions drain.

## Insert Rules

The GC-capable skiplist insert path should follow these rules:

- A writer may CAS through any non-frozen link.
- A writer must not CAS through a frozen link.
- A writer publishes new links with the active generation bit.
- A writer that sees a frozen or generation-mismatched link re-seeks from the
  active subversion head.
- Once level 0 insertion succeeds, higher-level failures due to frozen links do
  not need to abort the write. The node is already reachable at level 0.
- No overlapping GC passes are allowed on the same memtable.

The existing RocksDB write path already serializes memtable/SuperVersion
publication such that writers do not write to sealed memtables in old
`SuperVersion`s. This design relies on that discipline at the `MemTable*` level
and adds link-level protection for structure sharing below it.

## Why This Helps Mempurge

The current mempurge shape creates a new memtable-like object and then routes it
through immutable flush machinery. This design keeps mempurge inside the
memtable abstraction.

It should avoid or reduce issues around:

- immutable memtable list reordering
- `GetEarliestMemTableID()` and `GetLatestMemTableID()`
- manual flush waiting on mempurge replacement immutable memtables
- L0 file-number ordering caused by delayed mempurge output flush
- flush picking policy interactions

The logical memtable remains the same flush unit. GC changes its internal
representation, not its position in the flush pipeline.

## Implementation Touchpoints

Likely affected areas:

- `memtable/inlineskiplist.h`: tagged link representation and CAS rules
- `memtable/skiplistrep.cc`: GC-capable skiplist representation
- `db/memtable.h`: memtable subversion ownership and visibility metadata
- `db/column_family.h`: `SuperVersion` association with a specific memtable
  subversion
- write path: ensure subversion publication happens at a point where
  `MemTable*` ownership is already well defined

This should probably be introduced as a dedicated experimental memtable rep
rather than changing the ordinary skiplist rep for all users.

## Main Risks

- Tagged pointer portability and alignment assumptions.
- Ensuring all insert retry paths re-seek from the active subversion head when
  required.
- Maintaining sorted skiplist reachability while migration is partial.
- Freezing and bypassing links consistently across all relevant heights.
- Ensuring old capped subversions cannot expose partial new batches.
- Proving even/odd generation reuse is safe under the existing write-path
  serialization assumptions.

## Open Questions

- Is one generation bit enough, or should the tagged link use more epoch bits?
- What exact point in the write path should publish a new memtable subversion?
- Should generation mismatch alone force re-seek, or is the frozen bit
  sufficient for correctness?
- How should range tombstones and merge operands participate in memtable GC?
- Should GC be limited initially to sealed immutable memtables, active mutable
  memtables, or both?

