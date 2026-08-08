"""
Cortex test suite — conftest.py

Shared fixtures for all test modules.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortex.graph.state import CortexState, RunStatus, Task

# ── State fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def base_state() -> CortexState:
    return CortexState(
        run_id="test-run-001",
        session_id="test-session-001",
        user_id="test-user",
        user_goal="Summarise the quarterly sales report",
    )


@pytest.fixture
def state_with_tasks(base_state: CortexState) -> CortexState:
    tasks = [
        Task(id="fetch_data", description="Retrieve sales data", tool="search_knowledge"),
        Task(id="analyse", description="Analyse trends", depends_on=["fetch_data"]),
        Task(id="summarise", description="Write summary", depends_on=["analyse"]),
    ]
    return base_state.model_copy(update={"tasks": tasks})


@pytest.fixture
def completed_state(state_with_tasks: CortexState) -> CortexState:
    tasks = [t.mark_completed(f"Result for {t.id}") for t in state_with_tasks.tasks]
    return state_with_tasks.model_copy(
        update={
            "tasks": tasks,
            "final_output": "Q3 sales increased 12% YoY driven by enterprise segment.",
            "status": RunStatus.COMPLETED,
        }
    )


# ── LLM mock ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm_response():
    """Returns a mock LiteLLM ModelResponse."""

    def _make(content: str):
        response = MagicMock()
        response.choices[0].message.content = content
        response.choices[0].message.tool_calls = None
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        return response

    return _make


@pytest.fixture
def mock_router(mock_llm_response):
    """Replace the router SINGLETON, not the `get_router` name.

    This fixture used to `patch("cortex.llm.router.get_router")`, which did
    nothing: every agent does `from cortex.llm.router import get_router` at
    module scope, so the name is already bound in `cortex.agents.planner`
    and friends before the patch is applied. Patching the source module
    rebinds a name nobody reads.

    The consequence was not a silent no-op - it was worse. The real router
    ran, checked the real Redis cost ledger, and four agent tests died with
    `ConnectionError: Error 111 connecting to localhost:6379`. The suite
    could not pass without live infrastructure, which is a large part of why
    it had never been run at all.

    Patching the singleton fixes it for every consumer regardless of how
    they imported `get_router`, because they all resolve through the same
    module-level `_router`. Patch the state, not the name.
    """
    import cortex.llm.router as router_module

    router = AsyncMock()
    router.complete = AsyncMock(
        return_value=mock_llm_response(
            '{"reasoning": "test", "tasks": [{"id": "t1", "description": "test task", '
            '"tool": null, "depends_on": []}]}'
        )
    )
    previous = router_module._router
    router_module._router = router
    try:
        yield router
    finally:
        router_module._router = previous


# ── MCP mock ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_mcp_client():
    with patch("cortex.mcp.client.get_mcp_client") as mock:
        client = AsyncMock()
        client.call_tool = AsyncMock(return_value={"result": "mock tool result"})
        client.get_tool_schemas = AsyncMock(return_value=[])
        mock.return_value = client
        yield client
