"""Tests for modular scorecards and schemas."""

from __future__ import annotations

import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORECARDS_DIR = REPO_ROOT / "docs" / "scorecards"

EXPECTED_SCORECARDS = (
    "01_segmentation.md",
    "02_background.md",
    "03_appearance_crops.md",
    "04_motion_metadata.md",
    "05_residuals.md",
)
VALID_VERDICTS = {"RETIRE", "ACTIVE_SEARCH", "SATISFIED_FREEZE"}


def test_scorecards_directory_and_template_exist() -> None:
    assert SCORECARDS_DIR.is_dir()
    template = SCORECARDS_DIR / "TEMPLATE.md"
    assert template.is_file()
    content = template.read_text(encoding="utf-8")
    assert "Module Scorecard:" in content
    assert "Triad Definitions" in content
    assert "Short Horizon" in content
    assert "Long Horizon" in content


@pytest.mark.parametrize("filename", EXPECTED_SCORECARDS)
def test_each_module_scorecard_has_required_fields_and_valid_verdict(filename: str) -> None:
    card_path = SCORECARDS_DIR / filename
    assert card_path.is_file(), f"Missing scorecard: {filename}"
    content = card_path.read_text(encoding="utf-8")

    # Check required headings/fields
    assert "Module Scorecard:" in content
    assert "Owner Lane" in content
    assert "Source Scope" in content
    assert "Input Artifact" in content
    assert "Output Artifact" in content
    assert "Current Verdict" in content
    assert "Triad Definitions" in content
    assert "**Null**" in content
    assert "**Current**" in content
    assert "**Oracle**" in content
    assert "Short Horizon" in content
    assert "Long Horizon" in content
    assert "Decision Rule & Next Action" in content

    # Check valid verdict
    verdict_match = re.search(r"Current Verdict\*\*:\s*([A-Z_]+)", content)
    assert verdict_match is not None, f"No verdict found in {filename}"
    verdict = verdict_match.group(1)
    assert verdict in VALID_VERDICTS, f"Invalid verdict {verdict} in {filename}"
