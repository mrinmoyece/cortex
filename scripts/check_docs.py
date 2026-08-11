#!/usr/bin/env python3
"""Check repository Markdown links and local heading anchors without dependencies."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
_LINK_RE = re.compile(r"!?\[[^\]]*]\((?P<target><[^>]+>|[^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)")
_REFERENCE_DEFINITION_RE = re.compile(
    r"^\s{0,3}\[(?!\^)(?P<label>[^\]]+)]:\s*(?P<target><[^>]+>|\S+)"
)
_REFERENCE_USAGE_RE = re.compile(r"!?\[(?P<text>[^\]]+)]\[(?P<label>[^\]]*)]")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_NON_SLUG_RE = re.compile(r"[^\w\- ]", re.UNICODE)


def _slug(text: str) -> str:
    normalized = _NON_SLUG_RE.sub("", text.strip().lower())
    return re.sub(r"\s+", "-", normalized)


def _anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    seen: defaultdict[str, int] = defaultdict(int)
    in_fence = False

    for line in path.read_text(encoding="utf-8").splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING_RE.match(line)
        if not match:
            continue
        base = _slug(match.group(1))
        suffix = seen[base]
        seen[base] += 1
        anchors.add(base if suffix == 0 else f"{base}-{suffix}")

    return anchors


def _markdown_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.md")
        if ".git" not in path.parts and ".venv" not in path.parts
    )


def _reference_label(text: str) -> str:
    return " ".join(text.casefold().split())


def _validate_target(
    *,
    root: Path,
    source: Path,
    line_number: int,
    raw_target: str,
    anchor_cache: dict[Path, set[str]],
) -> list[str]:
    errors: list[str] = []
    raw_target = raw_target.strip("<>")
    parsed = urlsplit(raw_target)
    if parsed.scheme or raw_target.startswith("//"):
        return errors

    relative_path, separator, fragment = raw_target.partition("#")
    target = source if not relative_path else (source.parent / unquote(relative_path))
    target = target.resolve()

    try:
        target.relative_to(root.resolve())
    except ValueError:
        return [f"{source.relative_to(root)}:{line_number}: link escapes repository: {raw_target}"]

    if not target.exists():
        return [f"{source.relative_to(root)}:{line_number}: missing target: {raw_target}"]

    if separator and fragment and target.suffix.lower() == ".md":
        anchors = anchor_cache.setdefault(target, _anchors(target))
        decoded_fragment = unquote(fragment).lower()
        if decoded_fragment not in anchors:
            errors.append(
                f"{source.relative_to(root)}:{line_number}: missing anchor "
                f"#{fragment} in {target.relative_to(root)}"
            )

    return errors


def check_docs(root: Path = ROOT) -> list[str]:
    """Return all broken local Markdown link and anchor diagnostics."""
    errors: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}

    for source in _markdown_files(root):
        in_fence = False
        definitions: set[str] = set()
        usages: list[tuple[str, int]] = []
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue

            definition = _REFERENCE_DEFINITION_RE.match(line)
            if definition:
                definitions.add(_reference_label(definition.group("label")))
                errors.extend(
                    _validate_target(
                        root=root,
                        source=source,
                        line_number=line_number,
                        raw_target=definition.group("target"),
                        anchor_cache=anchor_cache,
                    )
                )

            for usage in _REFERENCE_USAGE_RE.finditer(line):
                label = usage.group("label") or usage.group("text")
                usages.append((_reference_label(label), line_number))

            for match in _LINK_RE.finditer(line):
                errors.extend(
                    _validate_target(
                        root=root,
                        source=source,
                        line_number=line_number,
                        raw_target=match.group("target"),
                        anchor_cache=anchor_cache,
                    )
                )

        for label, line_number in usages:
            if label not in definitions:
                errors.append(
                    f"{source.relative_to(root)}:{line_number}: undefined reference link: [{label}]"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root to check (default: inferred from this script)",
    )
    args = parser.parse_args()
    errors = check_docs(args.root.resolve())
    if errors:
        print("\n".join(errors))
        return 1
    print("Documentation links and anchors are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
