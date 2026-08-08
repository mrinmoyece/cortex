"""
Cortex LangGraph orchestration graph.

Graph topology:
                         ┌──────────┐
              ┌──────────►  planner  │
              │          └────┬─────┘
              │               │ (tasks created)
              │          ┌────▼──────┐
              │     ┌────► executor  │
              │     │    └────┬──────┘
              │  (retry)      │ (task done / all tasks done)
              │     │    ┌────▼──────┐
              │     └────┤  critic   │
              │          └────┬──────┘
              │   (accepted)  │  (rejected + iterations left)
              │          ┌────▼──────┐
              └──────────┤  END      │
           (replan)      └───────────┘

Special edges:
  - AWAITING_HUMAN: graph suspends; human feedback injected externally
  - budget_exceeded / max_iterations: forced transition to FAILED
"""

from __future__ import annotations

import uuid
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from cortex.agents.critic import CriticAgent
from cortex.agents.executor import ExecutorAgent
from cortex.agents.memory_agent import MemoryAgent
from cortex.agents.planner import PlannerAgent
from cortex.config import settings
from cortex.graph.state import CortexState, RunStatus
from cortex.logging_config import get_logger
from cortex.obs.metrics import agent_run_duration, agent_runs_total, memory_consolidation_failures

logger = get_logger(__name__)


# ── Node functions ────────────────────────────────────────────────────────────


async def load_memory_node(state: CortexState) -> dict:
    """Retrieve relevant memory before planning begins."""
    agent = MemoryAgent()
    memory_context = await agent.retrieve(
        user_goal=state.user_goal,
        session_id=state.session_id,
        user_id=state.user_id,
        # Threaded from state, not defaulted. The state has carried
        # `tenant_id` since day one and the memory layer ignored it.
        tenant_id=state.tenant_id,
    )
    return {
        "memory_context": memory_context,
        "status": RunStatus.PLANNING,
    }


async def planner_node(state: CortexState) -> dict:
    """Decompose the user goal into an ordered task list."""
    agent = PlannerAgent()
    try:
        tasks = await agent.plan(state)
        return {
            "tasks": tasks,
            "status": RunStatus.EXECUTING,
            "iteration_count": state.iteration_count + 1,
        }
    except Exception as exc:
        logger.error("planner.failed", run_id=state.run_id, error=str(exc))
        return {"status": RunStatus.FAILED, "error": f"Planning failed: {exc}"}


async def executor_node(state: CortexState) -> dict:
    """Execute the next pending task using MCP tools."""
    agent = ExecutorAgent()
    pending = state.pending_tasks()

    if not pending:
        return {"status": RunStatus.CRITIQUING}

    # Find first task whose dependencies are all completed
    completed_ids = {t.id for t in state.completed_tasks()}
    ready = [t for t in pending if all(d in completed_ids for d in t.depends_on)]

    if not ready:
        logger.error("executor.deadlock", run_id=state.run_id, pending=[t.id for t in pending])
        return {"status": RunStatus.FAILED, "error": "Task dependency deadlock detected"}

    task = ready[0]

    try:
        updated_task, cost_delta = await agent.execute_task(task, state)
        updated_tasks = [updated_task if t.id == task.id else t for t in state.tasks]

        all_done = all(t.status.value in ("completed", "skipped", "failed") for t in updated_tasks)

        # Compile output from completed tasks if all done
        output = None
        if all_done:
            output = await agent.compile_output(updated_tasks, state)

        return {
            "tasks": updated_tasks,
            "final_output": output,
            "total_cost_usd": state.total_cost_usd + cost_delta,
            "status": RunStatus.CRITIQUING if all_done else RunStatus.EXECUTING,
        }

    except Exception as exc:
        logger.error("executor.task_failed", run_id=state.run_id, task_id=task.id, error=str(exc))
        failed_task = task.mark_failed(str(exc))
        updated_tasks = [failed_task if t.id == task.id else t for t in state.tasks]
        return {"tasks": updated_tasks}


async def critic_node(state: CortexState) -> dict:
    """Review the compiled output and decide: accept, reject, or escalate."""
    if not state.final_output:
        return {"status": RunStatus.FAILED, "error": "No output to critique"}

    agent = CriticAgent()
    critique = await agent.critique(state)
    updated_critiques = [*state.critique_results, critique]

    if critique.accepted:
        logger.info("critic.accepted", run_id=state.run_id, score=critique.score)
        return {
            "critique_results": updated_critiques,
            "status": RunStatus.COMPLETED,
        }

    if state.critique_iteration + 1 >= state.max_critique_iterations:
        logger.warning("critic.max_iterations_hit", run_id=state.run_id)
        # Don't fail — surface best effort output with low-confidence flag
        return {
            "critique_results": updated_critiques,
            "output_metadata": {"low_confidence": True, "critique_score": critique.score},
            "status": RunStatus.COMPLETED,
        }

    logger.info(
        "critic.rejected_replanning", run_id=state.run_id, iteration=state.critique_iteration
    )
    return {
        "critique_results": updated_critiques,
        "critique_iteration": state.critique_iteration + 1,
        "tasks": [],  # Clear tasks — planner will rebuild
        "final_output": None,
        "status": RunStatus.PLANNING,
    }


async def save_memory_node(state: CortexState) -> dict:
    """Persist important facts and this run summary to memory stores.

    Failures here are swallowed on purpose, and the asymmetry is the point.
    This node runs AFTER the answer exists. Consolidation writes to Redis
    and Qdrant; if either is briefly unavailable the run had previously
    raised out of the graph and the user lost an answer that had already
    been produced and paid for - to save a note for next time.

    Losing the memory write is a degradation. Losing the answer is a bug.
    The failure is logged and surfaced on the metric so it is not invisible,
    but it does not propagate.
    """
    try:
        await MemoryAgent().consolidate(state)
    except Exception as exc:
        logger.warning(
            "memory.consolidation_failed",
            run_id=state.run_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        memory_consolidation_failures.inc()
    return {"status": RunStatus.COMPLETED}


# ── Routing functions ─────────────────────────────────────────────────────────


def route_after_executor(
    state: CortexState,
) -> Literal["executor", "critic", "end_failed"]:
    if state.status == RunStatus.FAILED:
        return "end_failed"
    if state.status == RunStatus.CRITIQUING:
        return "critic"
    # Budget / iteration guard
    # `settings.max_cost_per_run_usd`, not a literal. This read `>= 2.0`,
    # which happens to equal the default - so the guard looked correct and
    # raising the configured ceiling changed nothing here. A budget control
    # that ignores its own configuration is the most expensive kind of bug
    # to discover in production.
    if (
        state.total_cost_usd >= settings.max_cost_per_run_usd
        or state.iteration_count >= state.max_iterations
    ):
        logger.warning("graph.budget_or_iterations_exceeded", run_id=state.run_id)
        return "end_failed"
    return "executor"


def route_after_critic(
    state: CortexState,
) -> Literal["planner", "save_memory", "end_failed"]:
    if state.status == RunStatus.FAILED:
        return "end_failed"
    if state.status == RunStatus.COMPLETED:
        return "save_memory"
    # Rejected — replan
    return "planner"


# ── Graph assembly ────────────────────────────────────────────────────────────


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(CortexState)

    # Nodes
    builder.add_node("load_memory", load_memory_node)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("critic", critic_node)
    builder.add_node("save_memory", save_memory_node)

    # Entry
    builder.set_entry_point("load_memory")

    # Edges
    builder.add_edge("load_memory", "planner")
    builder.add_edge("planner", "executor")

    builder.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "executor": "executor",
            "critic": "critic",
            "end_failed": END,
        },
    )

    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "planner": "planner",
            "save_memory": "save_memory",
            "end_failed": END,
        },
    )

    builder.add_edge("save_memory", END)

    # Compile with in-memory checkpointer (swap for Redis/Postgres in prod)
    checkpointer = MemorySaver()
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["critic"],  # allow human-in-the-loop review point
    )


# ── Run helper ────────────────────────────────────────────────────────────────


async def run_cortex(
    user_goal: str,
    *,
    user_id: str,
    session_id: str | None = None,
    context: dict | None = None,
    tenant_id: str = "default",
) -> CortexState:
    """
    Execute a full Cortex agent run.

    Returns the final CortexState. Callers should check state.status
    to determine if the run completed successfully.
    """
    run_id = str(uuid.uuid4())
    session_id = session_id or str(uuid.uuid4())

    initial_state = CortexState(
        run_id=run_id,
        session_id=session_id,
        user_id=user_id,
        tenant_id=tenant_id,
        user_goal=user_goal,
        context=context or {},
    )

    graph = build_graph()
    config = {"configurable": {"thread_id": run_id}}

    import time

    start = time.perf_counter()
    agent_runs_total.labels(status="started").inc()

    try:
        final = await graph.ainvoke(initial_state, config=config)
        elapsed = time.perf_counter() - start
        agent_run_duration.observe(elapsed)
        agent_runs_total.labels(status=final["status"]).inc()
        logger.info(
            "run.completed",
            run_id=run_id,
            status=final["status"],
            latency_s=round(elapsed, 2),
            cost_usd=final.get("total_cost_usd", 0),
        )
        return CortexState(**final)

    except Exception as exc:
        elapsed = time.perf_counter() - start
        agent_runs_total.labels(status="error").inc()
        logger.error("run.error", run_id=run_id, error=str(exc), latency_s=round(elapsed, 2))
        raise
