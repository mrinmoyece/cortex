# MCP Integration Guide

Cortex is MCP-native. Every capability is exposed as a Model Context Protocol tool so any MCP-compatible client can use Cortex without writing integration code.

## Why MCP

MCP is the protocol standardised by Anthropic and adopted by OpenAI, Microsoft, and major IDE vendors. It defines how AI clients discover and call tools. Building on MCP means Cortex tools work in Claude Desktop, VS Code Copilot, Cursor, and any agent framework that supports the protocol.

## Available Tools

| Tool | Description | Key Args |
|------|-------------|----------|
| `search_knowledge` | Hybrid RAG search over ingested documents | `query`, `top_k`, `source_filter` |
| `query_memory` | Retrieve episodic or semantic memories **for the calling principal** | `query`, `memory_type`, `limit` |
| `execute_code` | Unsandboxed Python subprocess. **Disabled by default** | `code`, `timeout_seconds` |
| `query_data` | Natural language to SQL (requires database config) | `natural_language_query`, `database_alias` |
| `synthesise` | Summarise / transform content using LLM | `content`, `instruction`, `output_format` |

## Connecting Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "cortex": {
      "command": "/path/to/your/venv/bin/cortex-mcp",
      "env": {
        "MCP_TRANSPORT": "stdio",
        "OPENAI_API_KEY": "sk-...",
        "SECRET_KEY": "your-32-plus-char-key"
      }
    }
  }
}
```

Use the **absolute path** to the `cortex-mcp` entry point that `pip install -e .`
creates. Claude Desktop does not inherit your shell's `PATH`, and it does not
read the repository's `.env`, so every setting the server needs has to be in
`env` here.

Restart Claude Desktop. You'll see Cortex tools in the tool panel.

## Connecting over HTTP (remote server)

Set `MCP_TRANSPORT=http` and the server listens on `MCP_PORT` (8001 by
default) at `/mcp`:

```bash
MCP_TRANSPORT=http cortex-mcp
# -> http://localhost:8001/mcp
```

This is what `docker compose` runs for the `cortex-mcp` service.

`MCP_TRANSPORT` accepts `stdio` (default), `http`, or `sse`. `sse` is the
legacy transport, kept for older clients; new clients should use `http`.

## Reaching tools over the API

`POST /api/v1/mcp/call` authenticates the caller, binds their identity to the
call, and dispatches to the same in-process tool the MCP server exposes. The
reachable set is `MCP_HTTP_TOOL_ALLOWLIST` — not "whatever is registered", so
adding a tool for a desktop client does not silently publish it on an HTTP
endpoint. Configuration refuses to load if the allowlist names a tool that
does not exist; the default named `summarize_document` for a while, and the
real tool is `synthesise`, so a working tool answered 404.

Failures are classified rather than flattened:

| Outcome | Status | Code |
|---|---|---|
| Tool not in the allowlist | 404 | — |
| Argument the tool does not accept, or a missing required one | 422 | `MCP_TOOL_BAD_ARGUMENTS` |
| No identity bound for a tool that needs one | 403 | `MCP_FORBIDDEN` |
| The tool itself failed | 500 | `MCP_TOOL_FAILED` |

Argument binding is checked against the signature *before* the tool runs. A
`TypeError` raised inside a tool body is a bug in the tool and stays a 500 —
once it has propagated, it is indistinguishable from a bad call, which is why
the previous code could not tell them apart and returned 500 for both.

> **The HTTP transport has no authentication of its own.** FastMCP serves it
> unauthenticated, so binding it to anything other than a private network or
> a loopback interface publishes every tool — including memory reads — to
> whoever can reach the port. Put it behind an authenticating proxy, or go
> through the Cortex API's `POST /api/v1/mcp/call`, which authenticates the
> caller and binds the tool call to their identity.

## Tool Reference

### search_knowledge

Searches the RAG knowledge base. Uses hybrid dense+sparse retrieval with Cohere reranking.

```python
results = await search_knowledge(
    query="quarterly revenue growth drivers",
    top_k=5,
    source_filter="annual-report-2025"   # optional
)
# Returns: [{"content": "...", "score": 0.91, "source": "...", "retrieval_method": "reranked"}]
```

**When to use:** Any question answerable from ingested documents. Prefer this over synthesise for factual lookups.

### query_memory

Retrieves memories from episodic (past runs) or semantic (extracted facts) stores.

```python
memory = await query_memory(
    query="previous analysis of Q3 data",
    memory_type="all",   # "episodic" | "semantic" | "all"
    limit=5              # clamped to 1..50
)
# Returns: {"episodic": [...], "semantic": [...]}
```

**There is deliberately no `user_id` argument.** It used to take one, which
meant the *model* chose whose memory to read — a prompt injection, or simply
a guessed identifier, read another user's history. The identity now comes
from the authenticated principal:

```python
from cortex.mcp.server import Principal, use_principal

with use_principal(Principal(user_id="user-123", tenant_id="acme")):
    memory = await query_memory(query="previous analysis of Q3 data")
```

Calling a memory tool with no bound principal is refused. It fails closed on
purpose: defaulting to an anonymous or "system" identity would make every
unbound call read one shared store, which is the behaviour this replaced.

### Where the identity comes from, per entry point

| Entry point | Identity | Notes |
|---|---|---|
| `POST /api/v1/mcp/call` | The caller's JWT (`sub`, `tenant`) | Bound automatically |
| Agent graph (`ExecutorAgent`) | `CortexState.user_id` / `tenant_id`, from the request that created the run | Bound automatically |
| `cortex-mcp` over **stdio** | `MCP_PRINCIPAL_USER_ID`, if the operator sets it | One process, one declared user |
| `cortex-mcp` over **http**/**sse** | None. `query_memory` refuses | See below |

**The standalone server has no authentication, and none is invented for it.**
Over stdio that is workable: the client spawns the process, one process
serves one desktop client belonging to one person, and whoever writes that
client's config is in a position to say whose data it may touch.
`MCP_PRINCIPAL_USER_ID` is that statement, made explicitly, where the process
is launched:

```json
{
  "mcpServers": {
    "cortex": {
      "command": "/path/to/your/venv/bin/cortex-mcp",
      "env": {
        "MCP_TRANSPORT": "stdio",
        "MCP_PRINCIPAL_USER_ID": "your-user-id",
        "MCP_PRINCIPAL_TENANT_ID": "your-tenant",
        "OPENAI_API_KEY": "sk-...",
        "SECRET_KEY": "your-32-plus-char-key"
      }
    }
  }
}
```

Over **http** and **sse** it is not workable, and no configuration makes it
so: FastMCP serves every caller on the port identically, so a single declared
identity would hand one user's memory to whoever reached the socket. The
setting is therefore **ignored on those transports** — the server logs
`mcp.default_principal_refused` and `query_memory` keeps refusing. Route
network callers through `POST /api/v1/mcp/call`, which authenticates them
first.

A per-call binding always wins over the process-wide default, so a request
that does carry an identity is never served under someone else's.

**When to use:** Before starting a task, check if you've already done something similar for this user.

### execute_code

Runs Python in a subprocess. **This is not a sandbox**, and it is disabled
unless `CODE_EXECUTION_ENABLED=true` is set.

```python
result = await execute_code(
    code="""
import math
data = [1, 4, 9, 16, 25]
print([math.sqrt(x) for x in data])
""",
    timeout_seconds=10
)
# Returns: {"stdout": "[1.0, 2.0, 3.0, 4.0, 5.0]\n", "stderr": "", "exit_code": 0, "timed_out": False}
```

**Security:** the subprocess runs as the same OS user, on the same
filesystem, with the same network access and the same credentials as the API
process. The pattern denylist (`import os`, `import subprocess`, `import
sys`, `exec()`, `eval()`) stops a careless prompt, not an attacker —
`importlib`, `open`, `builtins` and `getattr` chains all walk straight past
it. Execution time is capped at 60s and output at a fixed size, which bounds
accidents, not intent.

Enable it only where running arbitrary attacker-supplied code is already an
accepted risk, or put a real isolation boundary (gVisor, Firecracker, a
separate execution service) underneath it first. When it is disabled the tool
is still registered — it returns a refusal rather than vanishing, so a client
that calls it gets an explanation instead of "unknown tool". See
[LIMITATIONS.md](LIMITATIONS.md).

### synthesise

General-purpose content transformation using the LLM.

```python
result = await synthesise(
    content="[long report text]",
    instruction="Extract the 5 most important findings as bullet points",
    output_format="markdown"
)
# Returns: {"result": "- Finding 1\n- Finding 2\n..."}
```

## Extending with New Tools

Add a new tool by decorating a function with `@mcp.tool()` in `src/cortex/mcp/server.py`:

```python
@mcp.tool()
async def fetch_github_issues(
    repo: str,
    state: str = "open",
    limit: int = 10,
) -> list[dict]:
    """
    Fetch GitHub issues for a repository.
    
    Args:
        repo:  GitHub repo in owner/name format (e.g. "anthropics/mcp")
        state: "open", "closed", or "all"
        limit: Maximum issues to return
    """
    # implementation
    ...
```

The `@mcp.tool()` decorator automatically:
- Generates the JSON schema from type hints and docstring
- Registers the tool for client discovery
- Makes it callable from the MCPClient in the executor agent

## Testing MCP Tools

```bash
# Run just the MCP tool tests
pytest tests/test_mcp/ -v

# Test a tool directly. Memory tools need a bound principal.
python -c "
import asyncio
from cortex.mcp.server import Principal, query_memory, use_principal

async def main():
    with use_principal(Principal(user_id='user-123', tenant_id='acme')):
        print(await query_memory(query='anything'))

asyncio.run(main())
"
```
