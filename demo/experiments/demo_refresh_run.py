"""Run the existing export path with bounded native CPU parallelism."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    env = dict(os.environ)
    env.update(
        PS_NATIVE_THREADS="8",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        FFMPEG="/opt/local/bin/ffmpeg",
    )
    result = subprocess.run(
        [sys.executable, "-m", "demo.experiments.export_demo_clips", *sys.argv[1:]],
        env=env,
        check=False,
    )
    if result.returncode:
        raise SystemExit(result.returncode)
    manifest = sys.argv[sys.argv.index("--source-manifest") + 1]
    raise SystemExit(
        subprocess.run(
            [
                sys.executable,
                "-m",
                "demo.experiments.validate_demo_refresh",
                "--source-manifest",
                manifest,
            ],
            env=env,
            check=False,
        ).returncode
    )


if __name__ == "__main__":
    main()
