#!/usr/bin/env python3
"""Project-scoped dependency audit.

`pip-audit --strict` audits the whole environment. On any machine that has
other projects installed - a developer laptop, a shared runner - that means
Cortex's build fails for a vulnerability in something Cortex does not
depend on, and passes only by luck when it does not. It also aborts on any
editable install it cannot resolve, which has nothing to do with Cortex.

This resolves the actual dependency closure of the `cortex` distribution
(optionally including an extra), pins each resolved package to the version
that is installed, and audits exactly that list.

Usage:
    python scripts/audit.py                # runtime dependencies, as installed
    python scripts/audit.py --extra dev    # plus the dev extra
    python scripts/audit.py --print        # show the closure, do not audit
    python scripts/audit.py --fresh        # audit what a clean install resolves

`--fresh` asks pip to resolve the tree from the index without touching the
environment, which is what CI actually installs. On a shared or long-lived
developer machine the installed versions can be years older than what
`pip install .` would pick today, and auditing those reports advisories
nobody can act on from this repository.

Ignored advisories live in `audit-ignore.toml` next to this script's repo
root, each with a reason and a review date, so an exception is a decision
someone made rather than a flag someone added.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

try:
    from packaging.markers import UndefinedEnvironmentName
    from packaging.requirements import Requirement
except ImportError:  # pragma: no cover - packaging ships with pip
    print("packaging is required: pip install packaging", file=sys.stderr)
    raise SystemExit(2) from None

ROOT = Path(__file__).resolve().parent.parent
IGNORE_FILE = ROOT / "audit-ignore.toml"


def _wanted(req: Requirement, extras: set[str]) -> bool:
    """Whether this requirement applies to the current interpreter."""
    if req.marker is None:
        return True
    candidates = extras | {""}
    for extra in candidates:
        try:
            if req.marker.evaluate({"extra": extra}):
                return True
        except UndefinedEnvironmentName:
            continue
    return False


def closure(root: str, extras: set[str]) -> dict[str, str]:
    """Resolve installed name==version for the whole dependency tree.

    A package is visited once *per set of extras*, not once. The previous
    version keyed `seen` on the bare name, so whichever requirement happened
    to be popped first decided the whole subtree: with `redis` reached
    through `celery[redis]` before `redis[hiredis]` from this project's own
    dependencies, the extras-qualified edge was discarded and **hiredis
    never entered the closure**. It was audited only by the accident of a
    LIFO queue ordering, and any new dependency on plain `redis` would have
    silently dropped it - the worst kind of gap in a security gate, because
    the gate keeps passing.
    """
    resolved: dict[str, str] = {}
    #: package key -> extras already expanded for it
    visited: dict[str, set[str]] = {}
    queue: list[tuple[str, set[str]]] = [(root, extras)]

    while queue:
        name, name_extras = queue.pop()
        key = name.lower().replace("_", "-")
        already = visited.get(key)
        if already is not None and name_extras <= already:
            continue
        visited[key] = (already or set()) | name_extras

        try:
            dist = distribution(name)
        except PackageNotFoundError:
            print(f"warning: {name} is not installed - skipping", file=sys.stderr)
            continue

        if key != root.lower().replace("_", "-"):
            resolved[dist.metadata["Name"]] = dist.version

        for raw in dist.requires or []:
            req = Requirement(raw)
            if not _wanted(req, name_extras):
                continue
            queue.append((req.name, set(req.extras)))

    return resolved


def fresh_closure(extras: set[str]) -> dict[str, str]:
    """Resolve the tree from the package index without installing anything."""
    target = f".[{','.join(sorted(extras))}]" if extras else "."
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "resolution.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--ignore-installed",
                "--quiet",
                "--report",
                str(report),
                target,
            ],
            cwd=ROOT,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit("pip could not resolve the dependency tree")
        data = json.loads(report.read_text())

    resolved: dict[str, str] = {}
    for item in data["install"]:
        meta = item["metadata"]
        if meta["name"].lower() == "cortex":
            continue
        resolved[meta["name"]] = meta["version"]
    return resolved


def load_ignores() -> list[str]:
    if not IGNORE_FILE.exists():
        return []
    data = tomllib.loads(IGNORE_FILE.read_text())
    ids: list[str] = []
    for entry in data.get("ignore", []):
        vuln_id = entry.get("id")
        if not vuln_id:
            raise SystemExit(f"{IGNORE_FILE}: every ignore entry needs an `id`")
        if not entry.get("reason") or not entry.get("review_by"):
            raise SystemExit(f"{IGNORE_FILE}: {vuln_id} needs a `reason` and a `review_by` date")
        ids.append(vuln_id)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra", action="append", default=[], help="include an optional extra")
    parser.add_argument("--print", action="store_true", help="print the closure and exit")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="resolve from the index instead of auditing installed versions",
    )
    args = parser.parse_args()

    extras = set(args.extra)
    packages = fresh_closure(extras) if args.fresh else closure("cortex", extras)
    lines = sorted(f"{name}=={version}" for name, version in packages.items())

    if args.print:
        print("\n".join(lines))
        return 0

    requirements = ROOT / ".audit-requirements.txt"
    requirements.write_text("\n".join(lines) + "\n")

    cmd = [sys.executable, "-m", "pip_audit", "-r", str(requirements), "--strict", "--no-deps"]
    for vuln_id in load_ignores():
        cmd += ["--ignore-vuln", vuln_id]

    print(f"auditing {len(lines)} packages in the cortex closure", file=sys.stderr)
    try:
        return subprocess.call(cmd)
    finally:
        requirements.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
