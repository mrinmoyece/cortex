# ADR 002: Use typed explicit state for the agent graph

- **Status:** Accepted
- **Date:** 2026-01
- **Owner:** `@mrinmoyece`

## Context

Planner, executor, critic, and memory nodes exchange identity, tasks, results,
cost, iteration, critique, and terminal status. Routing and checkpoint behavior
depend on those fields remaining consistent across loops and failures.

## Decision

Represent run state and nested task/critique records as Pydantic models. Nodes
return explicit updates, and routing functions decide from state status,
pending work, cost, and iteration counters.

Use LangGraph's in-memory checkpointer for the current reference
implementation. Durable checkpointing and API resume are not claimed.

## Consequences

- Validation and type information make graph transitions testable and
  inspectable.
- Cost and iteration values are visible to routing, but the current guard ends
  without first marking an executing state failed; this is incomplete
  conformance to the decision.
- Adding or changing persisted fields will require schema compatibility once a
  durable checkpointer exists.
- Current checkpoints do not survive restarts or move across replicas.
- Optional human review can expose a suspended state, but no repository API
  resumes it.

## Evidence

- [`src/cortex/graph/state.py`](../../src/cortex/graph/state.py)
- [`src/cortex/graph/cortex_graph.py`](../../src/cortex/graph/cortex_graph.py)
- [`tests/test_graph`](../../tests/test_graph)
- [Architecture: Agent graph and state](../ARCHITECTURE.md#agent-graph-and-state)
