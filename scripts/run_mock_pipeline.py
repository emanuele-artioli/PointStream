from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import run_mock_pipeline  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(run_mock_pipeline(), indent=2))
