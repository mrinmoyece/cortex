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

import time
import uuid
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from cortex.agents.critic import CriticAgent
from cortex.agents.executor import ExecutorAgent
from cortex.agents.memory_agent import MemoryAgent
from cortex.agents.planner import PlannerAgent
from cortex.config import settings
from cortex.graph.state import CortexState, MemoryContext, RunStatus
from cortex.logging_config import get_logger
from cortex.obs.metrics import (
    agent_iterations_per_run,
    agent_run_duration,
    agent_runs_total,
    agent_tasks_per_run,
    memory_consolidation_failures,
    memory_retrieval_failures,
    run_cost_usd,
)

logger = get_logger(__name__)


# ── Node functions ────────────────────────────────────────────────────────────


async def load_memory_node(state: CortexState) -> dict[str, Any]:
    """Retrieve relevant memory before planning begins.

    Memory is an enhancement, never a precondition. A Redis or Qdrant blip
    used to propagate out of this node and fail the run outright - the agent
    could still have answered the question, it just would not have had
    context. Degrading to an empty MemoryContext is the correct trade, and
    the failure is logged and counted rather than hidden.
    """
    agent = MemoryAgent()
    try:
        memory_context = await agent.retrieve(
            user_goal=state.user_goal,
            session_id=state.session_id,
            user_id=state.user_id,
            # Threaded from state, not defaulted. The state has carried
            # `tenant_id` since day one and the memory layer ignored it.
            tenant_id=state.tenant_id,
        )
    except Exception as exc:
        logger.warning(
            "memory.retrieval_failed",
            run_id=state.run_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        memory_retrieval_failures.inc()
        memory_context = MemoryContext()
    return {
        "memory_context": memory_context,
        "status": RunStatus.PLANNING,
    }


async def planner_node(state: CortexState) -> dict[str, Any]:
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


async def _run_total_cost(agent: Any, state: CortexState, cost_delta: float) -> float:
    """Best available figure for what this run has cost so far.

    The router's ledger is authoritative: the executor can make an extra LLM
    call after the task itself (output compilation), so the per-task delta
    alone under-reports. But the ledger lives in Redis, and a lookup failure
    there must not be able to reach the caller — it is raised from the same
    `try` that guards task execution, so an unhandled failure would mark a
    task that actually succeeded as failed and throw its result away.
    Accounting degrades to the locally accumulated total instead.
    """
    try:
        return float(await agent.get_run_cost(state.run_id))
    except Exception as exc:  # accounting must not be able to fail a run
        logger.warning("executor.cost_ledger_unavailable", run_id=state.run_id, error=str(exc))
        return state.total_cost_usd + cost_delta


async def executor_node(state: CortexState) -> dict[str, Any]:
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
            # The compiler can make an additional LLM call after the task,
            # so the ledger total is preferred over the task delta alone.
            "total_cost_usd": await _run_total_cost(agent, state, cost_delta),
            "status": RunStatus.CRITIQUING if all_done else RunStatus.EXECUTING,
        }

    except Exception as exc:
        logger.error("executor.task_failed", run_id=state.run_id, task_id=task.id, error=str(exc))
        failed_task = task.mark_failed(str(exc))
        updated_tasks = [failed_task if t.id == task.id else t for t in state.tasks]
        return {"tasks": updated_tasks}


async def critic_node(state: CortexState) -> dict[str, Any]:
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


async def save_memory_node(state: CortexState) -> dict[str, Any]:
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


def build_graph(*, human_review: bool | None = None) -> CompiledStateGraph[Any, Any, Any]:
    """Compile the agent graph.

    Args:
        human_review: Suspend the graph before the critic so a human can
            review the compiled output. Defaults to
            `settings.human_review_before_critic`.

            This is **off by default, deliberately**. It used to be
            unconditional, which meant every single `run_cortex` call
            suspended before the critic and returned a state that had never
            been critiqued and whose memory had never been saved - while the
            API reported the run as finished. A human-in-the-loop gate with
            nothing on the other side of it is not a review step, it is a
            silent truncation of every run. Turn it on only where something
            actually resumes the thread (see docs/LIMITATIONS.md).
    """
    if human_review is None:
        human_review = bool(settings.human_review_before_critic)

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
        interrupt_before=["critic"] if human_review else [],
    )


# ── Run helper ────────────────────────────────────────────────────────────────


def _status_label(status: RunStatus | str) -> str:
    """Metric label for a run status.

    `agent_runs_total.labels(status=state.status)` produced the label
    "RunStatus.COMPLETED" - `RunStatus` is a `(str, Enum)`, and prometheus
    stringifies the member, not its value. The literal "started" and "error"
    labels emitted elsewhere in this function therefore did not share a
    namespace with the terminal ones, and every dashboard filtering on
    `status="completed"` matched nothing.
    """
    return status.value if isinstance(status, RunStatus) else str(status)


async def run_cortex(
    user_goal: str,
    *,
    user_id: str,
    session_id: str | None = None,
    context: dict[str, Any] | None = None,
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
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    start = time.perf_counter()
    agent_runs_total.labels(status="started").inc()

    try:
        final = await graph.ainvoke(initial_state, config=config)
        elapsed = time.perf_counter() - start

        state = CortexState(**final)

        # A compiled-in interrupt returns a state that has not reached the
        # end of the graph. Reporting it as COMPLETED would be a lie, so the
        # suspension is made explicit in the returned status.
        snapshot = await graph.aget_state(config)
        if getattr(snapshot, "next", ()):
            state = state.model_copy(update={"status": RunStatus.AWAITING_HUMAN})
            logger.info(
                "run.suspended",
                run_id=run_id,
                next_nodes=list(snapshot.next),
                latency_s=round(elapsed, 2),
            )

        agent_run_duration.observe(elapsed)
        agent_runs_total.labels(status=_status_label(state.status)).inc()
        agent_tasks_per_run.observe(len(state.tasks))
        agent_iterations_per_run.observe(state.iteration_count)
        run_cost_usd.observe(state.total_cost_usd)
        logger.info(
            "run.completed",
            run_id=run_id,
            status=_status_label(state.status),
            latency_s=round(elapsed, 2),
            cost_usd=state.total_cost_usd,
        )
        return state

    except Exception as exc:
        elapsed = time.perf_counter() - start
        agent_runs_total.labels(status="error").inc()
        logger.error("run.error", run_id=run_id, error=str(exc), latency_s=round(elapsed, 2))
        raise
