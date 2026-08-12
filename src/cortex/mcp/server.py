"""
Cortex MCP Server.

Exposes Cortex capabilities as Model Context Protocol tools.
Any MCP-compatible client (Claude Desktop, VS Code Copilot, custom agents)
can connect to this server and call its tools.

Tools exposed:
  - search_knowledge   — hybrid RAG search over ingested documents
  - query_memory       — retrieve the calling principal's own memory
  - execute_code       — Python execution in a subprocess; NOT a sandbox,
                         and disabled unless CODE_EXECUTION_ENABLED=true
  - query_data         — natural-language-to-SQL over configured databases
  - synthesise         — structured summarisation / synthesis

There is no `web_search` tool. It was listed here and named in the planner's
prompt, but was never implemented, so the planner produced task plans
referencing a tool the executor could not call.

Run with:
    cortex-mcp
or:
    python -m cortex.mcp.server
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import sqlite3
import sys
import tempfile
import uuid
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# `fastmcp`, not `mcp.server.fastmcp`. Two different classes share the name:
# the one bundled in the `mcp` SDK takes no `version`, the standalone v2
# package does. This module imported the former and passed `version=` and
# `description=` to it, so `import cortex.mcp.server` raised TypeError on
# any machine with the declared dependencies installed. Both packages are in
# pyproject, which is what let the mistake look plausible.
from fastmcp import FastMCP

from cortex.agents.memory_agent import MemoryAgent
from cortex.config import settings
from cortex.logging_config import configure_logging, get_logger
from cortex.rag.pipeline import RAGPipeline

configure_logging()
logger = get_logger(__name__)

# FastMCP app — the MCP server instance
mcp = FastMCP(
    name="cortex",
    version="1.0.0",
    # `instructions`, not `description`: FastMCP v2 has no `description`
    # parameter, and this text is what a connecting client is shown.
    instructions=(
        "Cortex agentic AI platform - knowledge retrieval, long-term memory, "
        "and structured data queries."
    ),
)

#: Read-only SQLite databases this server may query, by alias. Empty by
#: default - a data tool that is reachable before anyone configured a
#: database is a tool that will be called and will fail.
DATABASE_ALIASES: dict[str, str] = {
    alias: path
    for alias, _, path in (
        entry.partition("=") for entry in (settings.sql_database_aliases or "").split(",") if entry
    )
    if alias and path
}

# ── Calling principal ─────────────────────────────────────────────────────────
#
# `query_memory` used to take `user_id` as a tool argument. Tool arguments are
# chosen by the model, and through `POST /api/v1/mcp/call` they are chosen
# directly by the HTTP caller - so "whose memory do I read" was an
# attacker-controlled string. Any authenticated user could read any other
# user's memory by passing their id, and a prompt-injection payload sitting
# in a retrieved document could make the model do it unprompted.
#
# The principal is now bound out of band, by whatever authenticated the
# request, and the tool cannot see or override it.

_principal: ContextVar[Principal | None] = ContextVar("cortex_mcp_principal", default=None)

#: Process-wide fallback identity, set only by `configure_default_principal`.
#: See `current_principal` for why this is not simply a default argument.
_default_principal: Principal | None = None


@dataclass(frozen=True)
class Principal:
    """The authenticated identity a tool call runs as."""

    user_id: str
    tenant_id: str = "default"


@contextlib.contextmanager
def use_principal(principal: Principal) -> Iterator[None]:
    """Bind the calling principal for the duration of a tool call."""
    token = _principal.set(principal)
    try:
        yield
    finally:
        _principal.reset(token)


def configure_default_principal(transport: str) -> Principal | None:
    """Install the process-wide identity for a standalone MCP server.

    An MCP transport carries no authentication. Over **stdio** that is fine
    to work with: the client spawns the process, one process serves one
    desktop client belonging to one person, and the operator writing that
    client's config is in a position to say whose data it may touch. Setting
    `MCP_PRINCIPAL_USER_ID` is that statement, made explicitly, in the place
    where the process is launched.

    Over **http** and **sse** it is not fine, and no configuration makes it
    fine: FastMCP serves every caller on the port identically, so a single
    declared identity would hand one user's memory to whoever reached the
    socket. This refuses to install a default there, whatever is configured,
    and says why. Reaching memory tools over the network goes through
    `POST /api/v1/mcp/call`, which authenticates the caller first.

    Returns the installed principal, or None.
    """
    global _default_principal

    configured = settings.mcp_principal_user_id
    if not configured:
        _default_principal = None
        logger.info("mcp.no_default_principal", transport=transport)
        return None

    if transport != "stdio":
        _default_principal = None
        logger.warning(
            "mcp.default_principal_refused",
            transport=transport,
            reason=(
                "MCP_PRINCIPAL_USER_ID is only honoured on the stdio transport. "
                "An HTTP/SSE MCP server has no per-caller authentication, so a "
                "process-wide identity would expose one user's data to every "
                "caller. Use POST /api/v1/mcp/call instead."
            ),
        )
        return None

    _default_principal = Principal(user_id=configured, tenant_id=settings.mcp_principal_tenant_id)
    logger.info(
        "mcp.default_principal_configured",
        transport=transport,
        tenant_id=_default_principal.tenant_id,
    )
    return _default_principal


def current_principal() -> Principal:
    """Return the bound principal, or refuse.

    Failing closed matters here: defaulting to an anonymous or "system"
    identity would make every memory tool call read some shared store, which
    is exactly the cross-tenant behaviour this replaced.

    The per-call binding wins over the process-wide default, so a request
    that *does* carry an identity is never served under someone else's.
    """
    principal = _principal.get() or _default_principal
    if principal is None:
        raise PermissionError(
            "No authenticated principal is bound for this MCP tool call, so "
            "tools that read user-owned data are unavailable. Call through "
            "POST /api/v1/mcp/call (which binds the caller's identity), run "
            "the tool inside cortex.mcp.server.use_principal(...), or - for a "
            "single-user stdio server only - set MCP_PRINCIPAL_USER_ID."
        )
    return principal


# Lazily initialised singletons (avoid slow startup on import)
_rag: RAGPipeline | None = None
_memory: MemoryAgent | None = None


def _get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
    return _rag


def _get_memory() -> MemoryAgent:
    global _memory
    if _memory is None:
        _memory = MemoryAgent()
    return _memory


# ── Tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
async def search_knowledge(
    query: str,
    top_k: int = 5,
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search the Cortex knowledge base using hybrid retrieval (dense + sparse + rerank).

    Args:
        query:         Natural language search query.
        top_k:         Number of results to return (1-20, default 5).
        source_filter: Optional — filter results by document source tag.

    Returns:
        List of ranked chunks with content, source, and relevance score.
    """
    top_k = max(1, min(top_k, 20))
    filters = {"source": source_filter} if source_filter else None

    rag = _get_rag()
    chunks = await rag.retrieve(query, top_k=top_k, filters=filters)

    return [
        {
            "content": c.content,
            "score": round(c.score, 4),
            "source": c.metadata.get("source", "unknown"),
            "chunk_index": c.metadata.get("chunk_index", 0),
            "retrieval_method": c.source,
        }
        for c in chunks
    ]


@mcp.tool()
async def query_memory(
    query: str,
    memory_type: str = "all",
    limit: int = 5,
) -> dict[str, Any]:
    """
    Retrieve relevant memories for the *calling user* from episodic or
    semantic stores.

    There is deliberately no `user_id` argument: the identity comes from the
    authenticated principal, never from the model or the caller.

    Args:
        query:       Query to match against stored memories.
        memory_type: "episodic" (past runs), "semantic" (facts), or "all".
        limit:       Maximum results per tier.

    Returns:
        Dict with "episodic" and/or "semantic" keys containing memory items.
    """
    principal = current_principal()
    limit = max(1, min(limit, 50))
    memory = _get_memory()
    result: dict[str, Any] = {}

    if memory_type in ("all", "episodic"):
        result["episodic"] = await memory.episodic.retrieve_recent(
            principal.user_id, limit=limit, tenant_id=principal.tenant_id
        )

    if memory_type in ("all", "semantic"):
        result["semantic"] = await memory.semantic.retrieve(
            query, principal.user_id, top_k=limit, tenant_id=principal.tenant_id
        )

    return result


@mcp.tool()
async def execute_code(
    code: str,
    language: str = "python",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """
    Execute Python in a subprocess. **Disabled by default.**

    This is not a sandbox, and the previous docstring calling it one was the
    most dangerous line in the repository. The subprocess runs as the same
    OS user, with the same filesystem, network and credentials as the API
    process; the pattern denylist below stops a careless prompt, not an
    attacker (`importlib`, `open`, `builtins`, and `getattr` chains all walk
    straight past it).

    It is therefore off unless `CODE_EXECUTION_ENABLED=true` is set
    deliberately, on a host where running arbitrary attacker-supplied code
    is already an accepted risk. See docs/LIMITATIONS.md.

    Args:
        code:             Python code to execute.
        language:         Must be "python" (other languages not supported).
        timeout_seconds:  Max execution time (1-60 seconds).

    Returns:
        Dict with "stdout", "stderr", "exit_code", and "timed_out" fields.
    """
    if not settings.code_execution_enabled:
        return {
            "error": (
                "Code execution is disabled. It is not sandboxed; set "
                "CODE_EXECUTION_ENABLED=true only on a host where arbitrary "
                "code execution is acceptable."
            ),
            "exit_code": -1,
        }

    if language != "python":
        return {
            "error": f"Language '{language}' not supported. Only 'python' is allowed.",
            "exit_code": -1,
        }

    timeout_seconds = max(1, min(timeout_seconds, 60))

    # Safety: block obvious dangerous imports
    dangerous = ["import os", "import subprocess", "import sys", "__import__", "exec(", "eval("]
    code_lower = code.lower()
    for pattern in dangerous:
        if pattern in code_lower:
            return {
                "error": f"Blocked: code contains disallowed pattern '{pattern}'",
                "exit_code": -1,
            }

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            return {
                "stdout": stdout.decode("utf-8", errors="replace")[:10_000],
                "stderr": stderr.decode("utf-8", errors="replace")[:2_000],
                "exit_code": proc.returncode,
                "timed_out": False,
            }
        except asyncio.TimeoutError:
            proc.kill()
            # Reaped, not just signalled. `kill()` without a `wait()` leaves
            # a zombie for the lifetime of the server process, so a handful
            # of timeouts exhausts the process table.
            with contextlib.suppress(Exception):
                await proc.wait()
            return {
                "stdout": "",
                "stderr": "Execution timed out",
                "exit_code": -1,
                "timed_out": True,
            }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


#: SQL a read-only tool is permitted to run. Deliberately an ALLOWLIST of
#: one statement type rather than a denylist of dangerous keywords: denylists
#: lose to `DELETE/**/FROM`, to `;`-chaining, and to whatever syntax the next
#: database version adds. Anything that is not a single bare SELECT is refused.
_SELECT_ONLY = re.compile(r"^\s*select\s", re.IGNORECASE)
_STATEMENT_SEPARATOR = re.compile(r";\s*\S")
MAX_SQL_ROWS = 500


@mcp.tool()
async def query_data(
    natural_language_query: str,
    database_alias: str = "default",
) -> dict[str, Any]:
    """
    Execute a natural-language query against a configured SQL database.

    Cortex translates the query to SQL, executes it read-only, and returns
    structured results.

    Args:
        natural_language_query: Plain English description of the data you need.
        database_alias:         Which configured database to query (default: "default").

    Returns:
        Dict with "sql" (generated query), "columns", "rows", and "row_count".
    """
    # This was a stub returning a placeholder while advertising itself to the
    # model as a working tool - so the agent would "query data", receive an
    # empty result set, and report confidently that there were no matching
    # records. A tool that lies about its own success is worse than one that
    # is absent, because the absent one cannot be planned around.
    path = DATABASE_ALIASES.get(database_alias)
    if path is None:
        return {
            "error": (
                f"Unknown database alias {database_alias!r}. "
                f"Configured: {sorted(DATABASE_ALIASES) or 'none'}"
            ),
            "row_count": 0,
        }

    schema = _describe_schema(path)
    if not schema:
        return {"error": f"Database {database_alias!r} has no readable tables.", "row_count": 0}

    sql = await _to_sql(natural_language_query, schema)

    # Two independent checks, because the SQL comes from a model and the
    # model is steerable by the documents it just read.
    if not _SELECT_ONLY.match(sql) or _STATEMENT_SEPARATOR.search(sql):
        logger.warning("query_data.rejected_sql", sql=sql[:200])
        return {
            "sql": sql,
            "error": "Refused: only a single SELECT statement may be executed.",
            "row_count": 0,
        }

    try:
        # `immutable=1` on a read-only URI: SQLite itself refuses writes, so
        # the guarantee does not rest solely on having parsed the SQL right.
        with contextlib.closing(
            sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        ) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(MAX_SQL_ROWS)
            columns = [d[0] for d in cursor.description or []]
    except sqlite3.Error as exc:
        return {"sql": sql, "error": f"Query failed: {exc}", "row_count": 0}

    return {
        "sql": sql,
        "columns": columns,
        "rows": [list(r) for r in rows],
        "row_count": len(rows),
        "truncated": len(rows) == MAX_SQL_ROWS,
    }


def _describe_schema(path: str) -> str:
    """Table and column names, so the model writes SQL against what exists."""
    try:
        with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            ]
            return "\n".join(
                f"{t}({', '.join(c[1] for c in conn.execute(f'PRAGMA table_info({t})'))})"
                for t in tables
            )
    except sqlite3.Error:
        return ""


async def _to_sql(question: str, schema: str) -> str:
    """Translate a question to SQL. Returns the bare statement."""
    from cortex.llm.router import get_router

    response = await get_router().complete(
        messages=[
            {
                "role": "system",
                "content": (
                    "Translate the question into ONE SQLite SELECT statement. "
                    "Return only SQL, no prose, no code fence, no trailing semicolon.\n\n"
                    f"Schema:\n{schema}"
                ),
            },
            {"role": "user", "content": question},
        ],
        run_id=str(uuid.uuid4()),
        temperature=0.0,
    )
    sql = (response.choices[0].message.content or "").strip()
    # Models fence SQL even when told not to.
    return re.sub(r"^```(?:sql)?|```$", "", sql, flags=re.MULTILINE).strip().rstrip(";")


@mcp.tool()
async def synthesise(
    content: str,
    instruction: str,
    output_format: str = "markdown",
) -> dict[str, Any]:
    """
    Synthesise, summarise, or transform a piece of content.

    Args:
        content:       The text to process.
        instruction:   What to do with it (e.g., "summarise in 3 bullets", "extract key entities").
        output_format: "markdown", "json", or "plain".

    Returns:
        Dict with "result" field containing the processed output.
    """
    from cortex.llm.router import get_router

    router = get_router()

    # Direct MCP calls are independent requests. A shared constant would
    # make unrelated callers consume each other's budget ledger.
    run_id = str(uuid.uuid4())
    messages = [
        {
            "role": "system",
            "content": f"You are a precise content synthesiser. Respond in {output_format} format only. No preamble.",
        },
        {
            "role": "user",
            "content": f"Instruction: {instruction}\n\nContent:\n{content[:8000]}",
        },
    ]

    response = await router.complete(messages=messages, run_id=run_id, temperature=0.0)
    return {"result": response.choices[0].message.content}


# ── Ingest endpoint (not a tool — used by the API layer) ─────────────────────


async def ingest_document(text: str, metadata: dict[str, Any] | None = None) -> int:
    """Called by the REST API to add documents to the knowledge base."""
    rag = _get_rag()
    return await rag.ingest(text, metadata)


# ── Server entrypoint ─────────────────────────────────────────────────────────


def run() -> None:
    """Run the Cortex MCP server.

    `import mcp` used to sit at the top of this function, rebinding the name
    of the module-level `FastMCP` instance to the `mcp` SDK package. The very
    next line then called `mcp.run(...)` on the package, which has no such
    attribute - so the `cortex-mcp` console script raised AttributeError on
    every invocation and the MCP server could not be started at all.
    """
    transport = str(settings.mcp_transport)
    configure_default_principal(transport)
    logger.info(
        "mcp.server_starting",
        transport=transport,
        host=settings.mcp_host,
        port=settings.mcp_port,
    )
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "http":
        mcp.run(transport="http", host=settings.mcp_host, port=settings.mcp_port)
    else:
        mcp.run(transport="sse", host=settings.mcp_host, port=settings.mcp_port)


if __name__ == "__main__":
    run()
