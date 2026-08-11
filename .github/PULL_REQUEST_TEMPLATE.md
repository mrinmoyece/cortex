<!--
Keep this short. The point is to make the change reviewable, not to fill in
a form. Delete any section that does not apply.
-->

## What this changes

<!-- One or two sentences. What behaviour is different after this merges? -->

## Why

<!-- The problem, not the patch. If it fixes an issue, link it. -->

## How it was verified

<!-- Which gates you actually ran, and anything you could not verify. -->

- [ ] `ruff check src tests perf scripts` and `ruff format --check src tests perf scripts`
- [ ] `mypy src`
- [ ] `pytest` (coverage gate is 80%)
- [ ] `bandit -r src -ll`
- [ ] `python3 scripts/audit.py --extra dev --fresh`
- [ ] `python3 -m perf.benchmark --concurrency 6 --iterations 25` (only if the
      request path changed)

## Risk

<!--
Anything a reviewer should look at twice: a new dependency, a changed
default, a widened permission, state that outlives a request, or a claim in
the docs that this change makes true or false.
-->

- [ ] Changes a default in `src/cortex/config.py`
- [ ] Adds or removes a dependency
- [ ] Touches auth, tenancy or the safety layer
- [ ] Updates the docs that describe the behaviour it changes
