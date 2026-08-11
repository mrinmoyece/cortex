"""Tests for the dependency-free Markdown link checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_CHECK_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_docs.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("cortex_check_docs_script", _CHECK_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def test_valid_relative_links_and_anchors_pass(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "README.md").write_text(
        "# Home\n\nSee [design](docs/design.md#data-flow).\n",
        encoding="utf-8",
    )
    (docs / "design.md").write_text("# Design\n\n## Data flow\n", encoding="utf-8")

    assert checker.check_docs(tmp_path) == []


def test_missing_file_is_reported(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("[missing](docs/nope.md)\n", encoding="utf-8")

    assert checker.check_docs(tmp_path) == ["README.md:1: missing target: docs/nope.md"]


def test_missing_anchor_is_reported(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "# Home\n\n[missing](#not-here)\n",
        encoding="utf-8",
    )

    assert checker.check_docs(tmp_path) == ["README.md:3: missing anchor #not-here in README.md"]


def test_links_inside_fenced_code_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "# Home\n\n```markdown\n[example](not-a-real-file.md)\n```\n",
        encoding="utf-8",
    )

    assert checker.check_docs(tmp_path) == []


def test_reference_style_target_is_checked(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "# Home\n\nSee [operations][ops].\n\n[ops]: docs/missing.md\n",
        encoding="utf-8",
    )

    assert checker.check_docs(tmp_path) == ["README.md:5: missing target: docs/missing.md"]


def test_undefined_reference_usage_is_reported(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "# Home\n\nSee [operations][ops].\n",
        encoding="utf-8",
    )

    assert checker.check_docs(tmp_path) == ["README.md:3: undefined reference link: [ops]"]


def test_angle_bracket_destination_with_spaces_is_checked(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "# Home\n\nSee [missing](<docs/missing file.md>).\n",
        encoding="utf-8",
    )

    assert checker.check_docs(tmp_path) == ["README.md:3: missing target: docs/missing file.md"]
