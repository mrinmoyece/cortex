# ADR 001: Use MCP for tool exposure

- **Status:** Accepted
- **Date:** 2026-01
- **Owner:** `@mrinmoyece`

## Context

Cortex tools must be discoverable by agent clients while remaining callable
from the authenticated HTTP API and internal graph. A bespoke REST contract
would require separate schemas and adapters for MCP-capable clients.

## Decision

Define tool implementations once and expose them through MCP discovery. Use an
in-process client for the graph and an allowlisted FastAPI adapter for
authenticated network calls.

MCP transport is not treated as authentication. Standalone stdio trusts its
process launcher; standalone HTTP/SSE has no repository-provided caller
identity. Per-user network tool access goes through
`POST /api/v1/mcp/call`, where JWT claims are bound to a `Principal`.

## Consequences

- Tool schemas and implementations remain shared across clients and the graph.
- MCP-compatible desktop clients can use stdio without a custom adapter.
- The API adapter must validate arguments, restrict exposed tool names, and
  preserve principal binding.
- Standalone MCP HTTP/SSE must remain on a trusted network and cannot expose
  principal-dependent tools safely without a future authentication design.
- The application maintains two transport surfaces even though business logic
  is shared.

## Evidence

- [`src/cortex/mcp/server.py`](../../src/cortex/mcp/server.py)
- [`src/cortex/mcp/client.py`](../../src/cortex/mcp/client.py)
- [`src/cortex/mcp/catalog.py`](../../src/cortex/mcp/catalog.py)
- [`tests/test_mcp`](../../tests/test_mcp)
- [Threat model](../THREAT_MODEL.md#trust-boundaries)
