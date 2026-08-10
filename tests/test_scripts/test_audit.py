"""The dependency audit's closure resolver.

`scripts/audit.py` decides which packages the security gate looks at. A gap
in it is invisible: the gate keeps passing, on fewer packages than anyone
thinks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_AUDIT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit.py"


def _load_audit():
    spec = importlib.util.spec_from_file_location("cortex_audit_script", _AUDIT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


def _dist(name: str, version: str, requires: list[str]):
    return SimpleNamespace(
        metadata={"Name": name},
        version=version,
        requires=requires,
    )


@pytest.fixture
def fake_world(monkeypatch):
    """A dependency graph reproducing the ordering that hid hiredis.

    `cortex` depends on `redis[hiredis]` and on `celery`; `celery` depends on
    plain `redis`. The queue is LIFO, so `celery` is expanded first and plain
    `redis` is reached before the extras-qualified edge.
    """
    world = {
        "cortex": _dist("cortex", "0.1.0", ["redis[hiredis]>=5", "celery>=5"]),
        "celery": _dist("celery", "5.4.0", ["redis>=4"]),
        "redis": _dist("redis", "6.4.0", ['hiredis>=3.0.0; extra == "hiredis"']),
        "hiredis": _dist("hiredis", "3.4.1", []),
    }

    def fake_distribution(name: str):
        key = name.lower().replace("_", "-")
        if key not in world:
            raise audit.PackageNotFoundError(name)
        return world[key]

    monkeypatch.setattr(audit, "distribution", fake_distribution)
    return world


class TestClosureFollowsExtras:
    def test_an_extras_dependency_is_included_when_the_plain_package_is_seen_first(
        self, fake_world
    ):
        """The regression. Keyed on the bare name, `redis` was marked visited
        by the `celery` edge, so `redis[hiredis]` was skipped entirely and
        hiredis - a compiled C extension, exactly the kind of package
        advisories are written about - was never audited."""
        resolved = audit.closure("cortex", set())
        assert "hiredis" in resolved
        assert resolved["hiredis"] == "3.4.1"

    def test_the_root_itself_is_not_audited(self, fake_world):
        assert "cortex" not in audit.closure("cortex", set())

    def test_every_reachable_package_appears_once(self, fake_world):
        resolved = audit.closure("cortex", set())
        assert set(resolved) == {"celery", "redis", "hiredis"}

    def test_revisiting_terminates(self, fake_world):
        """A package reachable with several different extras is expanded once
        per extras set, so the traversal has to converge rather than loop."""
        fake_world["cortex"] = _dist(
            "cortex", "0.1.0", ["redis[hiredis]>=5", "redis>=5", "celery>=5"]
        )
        assert "hiredis" in audit.closure("cortex", set())

    def test_a_missing_package_is_skipped_not_fatal(self, fake_world, capsys):
        fake_world["cortex"] = _dist("cortex", "0.1.0", ["not-installed>=1", "celery>=5"])
        resolved = audit.closure("cortex", set())
        assert "celery" in resolved
        assert "not-installed" in capsys.readouterr().err


class TestRootExtras:
    def test_a_root_extra_pulls_in_its_dependencies(self, fake_world):
        fake_world["cortex"] = _dist("cortex", "0.1.0", ['pytest>=8; extra == "dev"', "celery>=5"])
        fake_world["pytest"] = _dist("pytest", "8.0.0", [])

        assert "pytest" not in audit.closure("cortex", set())
        assert "pytest" in audit.closure("cortex", {"dev"})


class TestRealClosure:
    """An executable regression check against the actual environment, so the
    guarantee is about this repository rather than about a fixture."""

    def test_hiredis_is_in_the_real_runtime_closure(self):
        resolved = {k.lower() for k in audit.closure("cortex", set())}
        assert "redis" in resolved, "the closure resolver found nothing - is cortex installed?"
        assert "hiredis" in resolved, (
            "pyproject declares redis[hiredis]; if hiredis is absent from the "
            "closure the audit is not covering it"
        )


class TestIgnorePolicy:
    def test_the_ignore_file_parses_and_every_entry_is_justified(self):
        """An ignore without a reason and a review date is a permanent silent
        exception, which is how a suppression outlives the problem."""
        audit.load_ignores()
