"""docs/experiments/: one file per item, and every link into it resolves."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ITEMS = DOCS / "experiments"
LINK = re.compile(r"\]\(([^)\s]+)\)")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")


def slug(text: str) -> str:
    """GitHub's heading anchor: lower case, punctuation dropped, spaces to hyphens."""
    out = []
    for ch in unicodedata.normalize("NFC", text.replace("`", "")).lower():
        if ch == " ":
            out.append("-")
        elif ch in "-_" or unicodedata.category(ch)[0] in "LN" or unicodedata.category(ch) == "Mn":
            out.append(ch)
    return "".join(out)


def anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fence = not fence
        match = None if fence else HEADING.match(line)
        if match:
            base = slug(match.group(1))
            n = seen.get(base, 0)
            seen[base] = n + 1
    return {base if i == 0 else f"{base}-{i}" for base, count in seen.items() for i in range(count)}


def markdown_files() -> list[Path]:
    return sorted([*ROOT.glob("*.md"), *DOCS.glob("*.md"), *ITEMS.glob("*.md")])


def test_item_files_have_the_header_and_entries() -> None:
    files = sorted(ITEMS.glob("*.md"))
    assert files
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("# Experiments: "), path.name
        assert any(line.startswith("### ") for line in lines), path.name
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*\.md", path.name), path.name


def test_the_index_holds_no_run_entries() -> None:
    text = (DOCS / "experiments.md").read_text(encoding="utf-8")
    runs = text.split("\n## Runs\n", 1)[1]
    assert "\n### " not in runs


def test_links_into_the_experiments_docs_resolve() -> None:
    broken = []
    for source in markdown_files():
        for target in LINK.findall(source.read_text(encoding="utf-8")):
            if re.match(r"^(https?:|mailto:)", target):
                continue
            path, _, fragment = target.partition("#")
            dest = (source.parent / path).resolve() if path else source
            inside = dest == DOCS / "experiments.md" or ITEMS in dest.parents or dest == ITEMS
            if not inside:
                continue
            if not dest.exists():
                broken.append(f"{source.relative_to(ROOT)}: {target} (no such file)")
            elif fragment and dest.is_file() and fragment not in anchors(dest):
                broken.append(f"{source.relative_to(ROOT)}: {target} (no such heading)")
    assert not broken, "\n".join(broken)
