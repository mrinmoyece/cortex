"""
Cortex MCP Server.

Exposes Cortex capabilities as Model Context Protocol tools.
Any MCP-compatible client (Claude Desktop, VS Code Copilot, custom agents)
can connect to this server and call its tools.

Tools exposed:
  - search_knowledge   — hybrid RAG search over ingested documents
  - query_memory       — retrieve from the three-tier memory system
  - execute_code       — sandboxed Python execution
  - query_data         — natural-language-to-SQL over configured databases
  - web_search         — SerpAPI-backed web search
  - synthesise         — structured summarisation / synthesis

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
        "sandboxed code execution and structured data queries."
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
    filters = (
        {"must": [{"key": "source", "match": {"value": source_filter}}]} if source_filter else None
    )

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
    user_id: str,
    memory_type: str = "all",
    limit: int = 5,
) -> dict[str, Any]:
    """
    Retrieve relevant memories for a user from episodic or semantic stores.

    Args:
        query:       Query to match against stored memories.
        user_id:     User whose memory to search.
        memory_type: "episodic" (past runs), "semantic" (facts), or "all".
        limit:       Maximum results per tier.

    Returns:
        Dict with "episodic" and/or "semantic" keys containing memory items.
    """
    memory = _get_memory()
    result: dict[str, Any] = {}

    if memory_type in ("all", "episodic"):
        result["episodic"] = await memory.episodic.retrieve_recent(user_id, limit=limit)

    if memory_type in ("all", "semantic"):
        result["semantic"] = await memory.semantic.retrieve(query, user_id, top_k=limit)

    return result


@mcp.tool()
async def execute_code(
    code: str,
    language: str = "python",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """
    Execute code in a sandboxed subprocess. Only Python is supported.

    Args:
        code:             Python code to execute.
        language:         Must be "python" (other languages not supported).
        timeout_seconds:  Max execution time (1-60 seconds).

    Returns:
        Dict with "stdout", "stderr", "exit_code", and "timed_out" fields.
    """
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
        run_id="mcp-query-data",
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

    # Use a stub run_id for MCP tool calls
    run_id = "mcp-synthesise"
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


async def ingest_document(text: str, metadata: dict | None = None) -> int:
    """Called by the REST API to add documents to the knowledge base."""
    rag = _get_rag()
    return await rag.ingest(text, metadata)


# ── Server entrypoint ─────────────────────────────────────────────────────────


def run() -> None:
    """Run the Cortex MCP server (stdio transport for local; SSE for remote)."""
    import mcp

    logger.info("mcp.server_starting", host=settings.mcp_host, port=settings.mcp_port)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
