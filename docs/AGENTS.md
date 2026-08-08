# Agent Design

Cortex has four agents arranged in a LangGraph state machine. Each agent has a single, well-defined responsibility.

## Agent Responsibilities

```
load_memory → PlannerAgent → ExecutorAgent → CriticAgent → save_memory
                  ↑               ↑               │
                  └───────────────────────────────┘
                         (replan if rejected)
```

### PlannerAgent (`agents/planner.py`)

**Job:** Decompose the user goal into an ordered, dependency-aware task list.

**Inputs from state:**
- `user_goal` — what the user asked for
- `memory_context.semantic` — relevant facts already known
- `critique_results` — suggestions from the previous plan (if replanning)

**Output:** List of `Task` objects with `id`, `description`, `tool` hint, `depends_on`

**Prompt design decisions:**
- JSON-only output (`response_format={"type": "json_object"}`) — eliminates prose parsing
- Temperature=0.0 — deterministic planning
- Max 8 tasks constraint — prevents runaway plans that waste budget
- Memory context injected as "already known — don't re-fetch" — avoids duplicate retrieval
- Critique context injected verbatim when replanning — planner incorporates specific feedback

**Failure modes:**
- Invalid JSON → `AgentPlanningError` raised, run transitions to FAILED
- Missing `tasks` key → `AgentPlanningError` raised
- Duplicate task IDs → deduplicated automatically with a suffix

---

### ExecutorAgent (`agents/executor.py`)

**Job:** Execute a single task from the plan using MCP tools.

**Inputs from state:**
- Current pending task
- Results from completed dependency tasks
- Available MCP tool schemas

**Execution loop:**
```
1. Build prompt: goal + task description + dependency results + tool hint
2. LLM call with tool schemas as `tools` parameter
3. If tool_calls in response: execute each tool, append result, loop (max 3 rounds)
4. If text response: extract result string, mark task completed
5. If "TASK_FAILED: reason" in response: mark task failed
6. After 3 rounds with no resolution: mark task failed
```

**Tool call handling:**
- Tool errors are caught and returned as structured dicts `{"success": false, "error": "..."}`
- The LLM sees the error and can try an alternative tool or approach
- Persistent tool failures escalate to task failure, never crash the graph

**Cost tracking:**
- `cost_delta` is returned alongside the updated task
- The graph accumulates this into `state.total_cost_usd`

---

### CriticAgent (`agents/critic.py`)

**Job:** Score the compiled output and decide: accept, reject with suggestions, or fail open.

**Scoring dimensions:**
| Dimension | What it measures | Accept threshold |
|-----------|-----------------|------------------|
| `faithfulness` | Every claim grounded in task results. No hallucinations. | ≥ 0.85 |
| `completeness` | Full user goal addressed. Nothing material missing. | — |
| `coherence` | Clear, well-structured, unambiguous. | — |
| `overall` | Weighted composite. | ≥ 0.80 |

**Fail-open design:**
If the critic's LLM call fails or returns unparseable JSON, the critic returns `accepted=True` with `score=0.5`. The run completes with `low_confidence=True` in metadata rather than failing entirely. This is intentional — a degraded answer is better than no answer for most use cases.

**Replanning:**
When rejected, the critic's `suggestions` list is injected into the planner prompt on the next iteration. The planner can see exactly what was wrong and adjust the plan.

**Max iterations:**
`max_critique_iterations` defaults to 3. After 3 rejections, the best-effort output is accepted with `low_confidence=True`. This prevents infinite loops when the system genuinely can't produce a high-quality answer.

---

### MemoryAgent (`agents/memory_agent.py`)

**Job:** Load relevant context at run start; persist useful information at run end.

**Retrieval (load_memory_node):**
- `episodic.retrieve_recent(user_id, limit=5)` — last 5 run summaries from Redis
- `semantic.retrieve(user_goal, user_id, top_k=10)` — semantically similar facts from Qdrant
- Both run in parallel with `asyncio.gather`

**Consolidation (save_memory_node):**
- Write run summary (goal, status, output snippet, cost) to episodic Redis store
- Extract up to 5 facts from completed task results → upsert to Qdrant
- Non-blocking: consolidation failures are logged but don't fail the run

**Fact extraction (current approach):**
Uses a simple heuristic: task description + result (truncated to 300 chars) as the fact content. For production, replace `_extract_facts()` with an LLM call that extracts structured entities and statements from the full output.

---

## Graph Routing Logic

Routing decisions are pure functions of `state.status`. They live in `graph/cortex_graph.py`:

```python
def route_after_executor(state):
    if state.status == FAILED: return "end_failed"
    if state.status == CRITIQUING: return "critic"
    if state.total_cost_usd >= budget or state.iteration_count >= max: return "end_failed"
    return "executor"           # More tasks pending

def route_after_critic(state):
    if state.status == FAILED: return "end_failed"
    if state.status == COMPLETED: return "save_memory"
    return "planner"            # Rejected — replan
```

Pure routing functions make the graph easy to test and reason about.

## Interrupt / Human-in-the-loop

The graph is compiled with `interrupt_before=["critic"]`. This means you can pause the graph before the critic runs, let a human review the executor's output, inject feedback into `state.human_feedback`, then resume.

```python
# Pause at critic checkpoint
config = {"configurable": {"thread_id": run_id}}
graph.invoke(state, config=config)

# Inject human feedback
graph.update_state(config, {"human_feedback": "The cost figures look wrong."})

# Resume from checkpoint
graph.invoke(None, config=config)
```

This is used for high-stakes workflows where a human needs to approve before the output reaches end users.

## Adding a New Agent

1. Create `src/cortex/agents/my_agent.py`
2. Write a node function: `async def my_agent_node(state: CortexState) -> dict`
3. Return a dict with only the state keys you want to update
4. Add the node to `build_graph()` in `graph/cortex_graph.py`
5. Wire conditional edges from the nodes that should route to it
6. Add tests in `tests/test_agents/`
