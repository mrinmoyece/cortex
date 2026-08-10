# Security Policy

## Supported versions

Cortex is a single-branch project. Only `main` receives fixes; there are no
maintained release branches and no backports.

## Reporting a vulnerability

Report privately through GitHub's
[security advisory form](https://github.com/mrinmoyece/cortex/security/advisories/new).
Please do not open a public issue for anything exploitable.

Include what you have: the affected file or endpoint, the conditions needed
to reach it, and the impact. A proof of concept helps but is not required.

This is a personal open-source project, not a vendored product. There is no
guaranteed response time and no bug bounty. Reports are read and acted on as
time allows, and credit is given in the advisory unless you prefer otherwise.

## Scope

In scope: anything in `src/cortex`, the deployment manifests under `deploy/`
and `docker-compose.yml`, and the CI workflows.

Out of scope, because they are known and documented rather than undiscovered
— see [docs/LIMITATIONS.md](docs/LIMITATIONS.md):

* **`execute_code` is not a sandbox.** It runs a subprocess in the same
  container as the API. It is disabled by default (`CODE_EXECUTION_ENABLED`)
  and the docstring says exactly this. Enabling it and then executing code is
  the documented behaviour, not a vulnerability.
* **Run state and rate limiting are per-process.** Both are in-memory, so
  they do not hold across replicas. This is stated in the limitations doc.
* **Default configuration is a local development configuration.** An ephemeral
  `SECRET_KEY`, `0.0.0.0` binds and an unauthenticated `/metrics` are defaults
  for `docker compose up`, and each one has a setting to change it.

## Handling of secrets

No credentials are committed to this repository. `deploy/k8s/service.yaml`
contains placeholder values (`REPLACE_WITH_...`) intended to be replaced by a
real secret manager; it is not a working secret. `.env.example` documents
variable names only.
