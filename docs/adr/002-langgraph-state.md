# ADR 002: LangGraph State Schema Design

**Status:** Accepted  
**Date:** 2026-01  

---

## Context

The agent orchestration graph passes a state object between nodes. The schema of that state object determines what information is available to each agent, how the graph makes routing decisions, and what gets checkpointed for resumability.

Key questions:
1. Single flat dict vs typed Pydantic model?
2. Mutable vs append-only history?
3. How granular should task tracking be?
4. Where do we track cost and iteration counts?

---

## Decision

State is a **Pydantic BaseModel (`CortexState`)** with:
- Typed, documented fields for every piece of state
- Append-only `messages` list (LangGraph `add_messages` annotation)
- Explicit `Task` objects with per-task status tracking
- Cost counter and iteration counter in the state (not external)

---

## Rationale

**Typed model over dict.** Untyped dicts make refactoring dangerous and agent bugs hard to diagnose. Pydantic models give us validation, IDE autocompletion, and self-documenting code. The overhead is negligible.

**Append-only messages.** LangGraph's `add_messages` annotation ensures message history is never accidentally overwritten by a node. This is critical for graph resumability — a replanning step should not lose the history of why the previous plan failed.

**Explicit Task objects.** Having tasks as first-class objects with status, result, and dependency fields allows the executor to implement dependency-aware scheduling without complex logic. The planner writes tasks; the executor ticks them off. Critique feedback maps cleanly to task-level failures.

**Cost and iteration in state.** Cost tracking needs to work across replanning loops (where the budget should be shared). Keeping these counters in state means the graph routing functions can enforce budget and iteration limits without consulting an external service.

---

## Consequences

**Positive:**
- Graph routing decisions are purely functions of state — easy to test.
- Checkpointing (LangGraph MemorySaver / PostgresSaver) works out of the box with Pydantic serialisation.
- Budget and iteration limits are enforced at the graph level, not inside agents.

**Negative:**
- Pydantic models are less flexible than dicts for schema evolution. Adding fields is fine; removing or renaming requires migrations if state is persisted.
- The state object grows with the conversation. For very long runs with many tasks, the checkpointed state can become large.

---

## Mitigations

- Schema changes follow a deprecation cycle: add new field, mark old field deprecated, remove after 2 releases.
- State compaction: after task completion, task results are summarised and the raw task list is cleared before storage.
