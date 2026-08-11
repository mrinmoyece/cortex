# Security Policy

## Supported versions

Cortex has no maintained release branches. Security fixes are made on `main`;
older commits and forks are not supported.

## Reporting a vulnerability

Report exploitable issues privately through the
[GitHub security advisory form](https://github.com/mrinmoyece/cortex/security/advisories/new).
Do not open a public issue. Include the affected entry point or file,
preconditions, impact, and a proof of concept when available.

This is a personal open-source project with no guaranteed response time or bug
bounty. Reports are handled as maintainer availability permits.

## Scope

Runtime code under [`src/cortex`](src/cortex), deployment artifacts under
[`deploy`](deploy) and [`docker-compose.yml`](docker-compose.yml), CI, and
repository automation are in scope.

Known and explicitly documented limitations are not undisclosed
vulnerabilities by themselves. Examples include disabled-by-default
unsandboxed code execution, process-local state, unauthenticated standalone
MCP network transports, and public metrics when no token or network control is
configured. Their exact boundaries and expected controls are documented in
the [threat model](docs/THREAT_MODEL.md) and
[limitations](docs/LIMITATIONS.md). A bypass of a stated control remains in
scope.

Do not commit credentials. [`.env.example`](.env.example) contains names and
development placeholders only; production secrets must be supplied by the
deployment environment.
