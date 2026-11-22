"""Retain the complete source-only denominator including any unresolved numerical gate."""

from pathlib import Path
import json

HERE = Path(__file__).resolve().parent


def run():
    rows = []
    for src in json.loads((HERE / "sources.json").read_text()):
        f = HERE / (src["tag"] + "-source-result.json")
        row = {k: src[k] for k in ["tag", "rank", "native_fps", "source_sha256"]}
        if not f.exists():
            row.update(status="source screen pending")
            rows.append(row)
            continue
        result = json.loads(f.read_text())
        assert (
            result["sha256"] == src["source_sha256"]
            and result["output_raster"] == [1920, 1080]
            and len(result["rows"]) == 240
        )
        count = sum(bool(v["fits"]) for v in result["rows"])
        assert result["eligible"] == (count >= 192)
        row.update(
            status="court eligible" if result["eligible"] else "conventional fallback",
            court_fits=count,
            frames=240,
            source_size=[result["source"]["width"], result["source"]["height"]],
            output_size=result["output_raster"],
            cpu_host_wall_sec=result["wall_seconds"],
        )
        rows.append(row)
    (HERE / "summary.json").write_text(
        json.dumps(
            dict(
                source_files=len(rows),
                decoded_source_ticks=sum(r.get("frames", 0) for r in rows),
                input_event_count=2,
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    run()
