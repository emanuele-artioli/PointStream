"""Reproduce reuse-cost arithmetic from published experiment receipts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECEIPT = ROOT / "docs/experiments/clean-court-grabcut-20261119/delivery-receipt.json"
HISTORICAL = ROOT / "docs/experiments/w9-inputs/w7-tier1-billing-corrected.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = json.loads(RECEIPT.read_text())["rows"]
    paid = next(row for row in rows if row["name"] == "grabcut")
    historical = json.loads(HISTORICAL.read_text())["windows"][0]
    assert historical["window"] == "al028"
    assert paid["total_bytes"] == 414081
    references = paid["components"]["ref0"]["bytes"] + paid["components"]["ref1"]["bytes"]
    assert references == 242317
    running = paid["total_bytes"] - references
    assert running == 171764
    historical_rate = historical["rate_kbps"]
    assert abs(historical_rate - 19.239833333333333) < 1e-10
    adapter = 1488252
    bank = 6 * 155683
    horizons = (4.8, 30, 60, 600, 1800, 3600, 7200)
    sensitivity = []
    for seconds in horizons:
        rally_rate = running * 8 / (4.8 * 1000) + references * 8 / (seconds * 1000)
        historical_modeled_rate = historical_rate + (adapter + bank) * 8 / 1000 * (1 / seconds - 1 / 7200)
        sensitivity.append({"reuse_seconds": seconds, "rally_kbps": rally_rate,
                            "historical_modeled_kbps": historical_modeled_rate})
    output = {
        "receipt_sha256": digest(RECEIPT),
        "historical_sha256": digest(HISTORICAL),
        "rally_total_bytes": paid["total_bytes"],
        "rally_reference_bytes": references,
        "rally_remaining_bytes": running,
        "rally_duration_seconds": 4.8,
        "historical_bank_bytes": bank,
        "historical_adapter_and_guider_bytes": adapter,
        "sensitivity": sensitivity,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
