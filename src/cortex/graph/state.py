"""
Cortex agent graph state.

The State object is the single source of truth passed between every
node in the LangGraph graph. Every field is typed and documented.
LangGraph serialises this to/from its checkpoint store automatically.

Design principles:
- Immutable history — append-only lists, never mutate past entries
- Explicit status tracking — no ambiguous boolean flags
- Cost and iteration counters for budget / loop-guard enforcement
- Structured task list — planner writes it, executor ticks it off
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

#: `datetime.UTC` is 3.11+; this package supports 3.10.
UTC = timezone.utc


class RunStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    CRITIQUING = "critiquing"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_HUMAN = "awaiting_human"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Task(BaseModel):
    id: str
    description: str
    tool: str | None = None  # MCP tool to use, if known
    depends_on: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def mark_started(self) -> Task:
        return self.model_copy(
            update={
                "status": TaskStatus.IN_PROGRESS,
                "started_at": datetime.now(UTC),
            }
        )

    def mark_completed(self, result: str) -> Task:
        return self.model_copy(
            update={
                "status": TaskStatus.COMPLETED,
                "result": result,
                "completed_at": datetime.now(UTC),
            }
        )

    def mark_failed(self, error: str) -> Task:
        return self.model_copy(
            update={
                "status": TaskStatus.FAILED,
                "error": error,
                "completed_at": datetime.now(UTC),
            }
        )


class CritiqueResult(BaseModel):
    accepted: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    suggestions: list[str] = Field(default_factory=list)
    iteration: int = 0


class MemoryContext(BaseModel):
    """Memory retrieved at run start, injected into agent prompts."""

    episodic: list[dict[str, Any]] = Field(default_factory=list)  # past runs
    semantic: list[dict[str, Any]] = Field(default_factory=list)  # facts & knowledge
    working: list[dict[str, Any]] = Field(default_factory=list)  # current context


class CortexState(BaseModel):
    """
    Full state object for an Cortex agent run.

    Passed through every node of the LangGraph graph.
    LangGraph handles serialisation and checkpointing.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    run_id: str
    session_id: str
    user_id: str
    tenant_id: str = "default"

    # ── Input ─────────────────────────────────────────────────────────────────
    user_goal: str
    context: dict[str, Any] = Field(default_factory=dict)

    # ── Conversation history (LangGraph managed) ──────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # ── Planning ──────────────────────────────────────────────────────────────
    tasks: list[Task] = Field(default_factory=list)
    current_task_id: str | None = None

    # ── Output ────────────────────────────────────────────────────────────────
    final_output: str | None = None
    output_metadata: dict[str, Any] = Field(default_factory=dict)

    # ── Critique loop ─────────────────────────────────────────────────────────
    critique_results: list[CritiqueResult] = Field(default_factory=list)
    critique_iteration: int = 0
    max_critique_iterations: int = 3

    # ── Memory ────────────────────────────────────────────────────────────────
    memory_context: MemoryContext = Field(default_factory=MemoryContext)

    # ── Status & control ─────────────────────────────────────────────────────
    status: RunStatus = RunStatus.PENDING
    error: str | None = None
    human_feedback: str | None = None

    # ── Accounting ────────────────────────────────────────────────────────────
    total_cost_usd: float = 0.0
    iteration_count: int = 0
    max_iterations: int = 25

    # ── Timestamps ───────────────────────────────────────────────────────────
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def pending_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def completed_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    def all_tasks_done(self) -> bool:
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.FAILED)
            for t in self.tasks
        )

    def last_critique(self) -> CritiqueResult | None:
        return self.critique_results[-1] if self.critique_results else None

    def touch(self) -> CortexState:
        return self.model_copy(update={"updated_at": datetime.now(UTC)})
