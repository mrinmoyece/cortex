# MCP Integration Guide

Cortex is MCP-native. Every capability is exposed as a Model Context Protocol tool so any MCP-compatible client can use Cortex without writing integration code.

## Why MCP

MCP is the protocol standardised by Anthropic and adopted by OpenAI, Microsoft, and major IDE vendors. It defines how AI clients discover and call tools. Building on MCP means Cortex tools work in Claude Desktop, VS Code Copilot, Cursor, and any agent framework that supports the protocol.

## Available Tools

| Tool | Description | Key Args |
|------|-------------|----------|
| `search_knowledge` | Hybrid RAG search over ingested documents | `query`, `top_k`, `source_filter` |
| `query_memory` | Retrieve episodic or semantic memories for a user | `query`, `user_id`, `memory_type` |
| `execute_code` | Sandboxed Python execution | `code`, `timeout_seconds` |
| `query_data` | Natural language to SQL (requires database config) | `natural_language_query`, `database_alias` |
| `synthesise` | Summarise / transform content using LLM | `content`, `instruction`, `output_format` |

## Connecting Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "cortex": {
      "command": "python",
      "args": ["-m", "cortex.mcp.server"],
      "cwd": "/path/to/cortex",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "SECRET_KEY": "your-32-char-key"
      }
    }
  }
}
```

Restart Claude Desktop. You'll see Cortex tools in the tool panel.

## Connecting via SSE (remote server)

For remote or Docker deployments, run the MCP server in SSE mode:

```bash
# In docker-compose, cortex-mcp runs on port 8001
# Configure your client to connect to:
http://localhost:8001/sse
```

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
    user_id="user-123",
    memory_type="all",   # "episodic" | "semantic" | "all"
    limit=5
)
# Returns: {"episodic": [...], "semantic": [...]}
```

**When to use:** Before starting a task, check if you've already done something similar for this user.

### execute_code

Runs Python in a subprocess sandbox. Blocks dangerous imports (os, subprocess, sys).

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

**Security:** Blocks `import os`, `import subprocess`, `import sys`, `exec()`, `eval()`. Max execution time 60s.

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

# Test a specific tool directly
python -c "
import asyncio
from cortex.mcp.server import execute_code
result = asyncio.run(execute_code(code='print(2+2)'))
print(result)
"
```
