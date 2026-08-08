# ADR 001: Use MCP over Custom REST for Tool Exposure

**Status:** Accepted  
**Date:** 2026-01  
**Deciders:** Cortex core team  

---

## Context

Cortex exposes capabilities (search, memory, code execution, data query) to external clients — both human-operated tools (Claude Desktop, VS Code) and automated agents. We needed to decide how to expose these capabilities.

Options considered:
1. Custom REST API with OpenAPI spec
2. gRPC with protobuf schemas
3. Model Context Protocol (MCP)

---

## Decision

We use **Model Context Protocol (MCP)** as the tool exposure standard.

---

## Rationale

**Native LLM client compatibility.** MCP is the protocol adopted by Anthropic, OpenAI (via adapters), and major IDE vendors. Any MCP-compatible client can connect to Cortex without custom integration code. This is an order of magnitude less friction than a bespoke REST API.

**Standardised tool discovery.** MCP clients automatically discover available tools, their schemas, and descriptions. No separate documentation step required for tool consumers.

**Future-proof.** The ecosystem around MCP is growing rapidly. Tools, clients, and server implementations are being standardised. Aligning with this now avoids a migration later.

**Composability.** Cortex can itself act as an MCP client and consume tools from other MCP servers (GitHub, Jira, Slack) without additional adapters.

---

## Consequences

**Positive:**
- Claude Desktop, VS Code Copilot, and any other MCP client can use Cortex tools out of the box.
- Tool schemas are self-documenting.
- We can consume other MCP servers natively.

**Negative:**
- MCP is not yet universally supported — some enterprise systems will still need REST adapters.
- MCP tooling (debugging, testing) is less mature than REST tooling.
- We maintain both the MCP server (for LLM clients) and a REST API (for human-facing UIs and non-MCP systems).

---

## Mitigations

- The FastAPI REST layer wraps the same underlying tool functions. There is no duplication of business logic — only the transport layer differs.
- We will add REST-to-MCP adapters for legacy enterprise integrations as needed.
