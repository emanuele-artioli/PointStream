"""Launch a bounded common-player worker with a read-only input allowlist on campus Linux."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import tempfile
from typing import Any

from experiments.tennis import common_receiver_manifest as manifest
from experiments.tennis import w8_cuda_system_leaves as policy
from experiments.tennis import w8_strict_source as code


def launch(
    *,
    source_root: Path,
    output: Path,
    worker_config: dict,
    inputs: list[Path],
    denied: list[Path],
    seconds: int,
    kind: str,
) -> dict:
    if (
        kind not in ("conditioning", "generation", "finish", "native", "mixed", "rung")
        or not 1 <= seconds <= 2700
    ):
        raise ValueError("unsupported worker or budget")
    expected = {"carrier", "manifest", "expected_sha256", "frames", "units"}
    required = [Path(worker_config["carrier"]), Path(worker_config["manifest"])]
    if kind == "generation":
        expected |= {"weights", "fit"}
        required += [
            Path(worker_config["fit"]),
            *(Path(v) for v in worker_config["weights"].values()),
        ]
    if kind in ("finish", "native", "mixed", "rung"):
        expected = {"carrier", "carrier_sha256", "manifest", "ffmpeg"}
        required += [Path(worker_config["ffmpeg"])]
    if kind in ("finish", "mixed"):
        expected |= {"sam", "generated", "generation_receipts", "ticks"}
        required += [
            Path(worker_config["sam"]),
            *(
                Path(r["path"])
                for k in ("generated", "generation_receipts")
                for r in worker_config[k].values()
            ),
        ]
    if kind == "mixed":
        expected |= {"joint", "cache"}
        required += [Path(worker_config["joint"]), Path(worker_config["cache"])]
    if kind == "rung":
        names = ("joint", "cache", "original_joint", "witness_frames", "witness_receipt")
        expected |= {*names, "masks", "ticks"}
        required += [
            *(Path(worker_config[k]) for k in names),
            *(Path(p) for p in worker_config["masks"].values()),
        ]
    if (
        set(worker_config) != expected
        or set(inputs) != set(required)
        or not all(p.is_file() for p in required)
    ):
        raise ValueError("worker configuration or exact input set differs")
    if len(set(denied)) < 1 or not all(p.is_file() for p in denied):
        raise ValueError("existing source/golden denial probes required")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    control = Path(tempfile.mkdtemp(prefix="receiver-control-", dir=output.parent))
    staged = control / "code"
    source_manifest = code.stage_source(source_root, staged)
    code.verify_source(staged, source_manifest)
    canary = control / "write-test"
    canary.write_bytes(b"owned receiver denial probe")
    work = control / "private"
    work.mkdir()
    config = dict(
        kind=kind,
        settings=worker_config,
        inputs=[str(p.resolve(strict=True)) for p in inputs],
        denied=[str(p.resolve(strict=True)) for p in denied],
        canary=str(canary),
        output=str(work),
        code=str(staged),
        seconds=seconds,
        code_manifest_sha256=source_manifest["manifest_sha256"],
        namespaces={name: os.readlink("/proc/self/ns/" + name) for name in ("net", "user")},
    )
    encoded = json.dumps(config, sort_keys=True).encode()
    env_sha = (
        manifest.SAM_ENV if kind in ("finish", "native", "mixed", "rung") else manifest.GEN_ENV
    )
    interpreter = Path("/local/users/emanuele/pointstream/cache") / env_sha / "tree/bin/python"
    env = {
        k: v
        for k, v in os.environ.items()
        if k in ("CUDA_VISIBLE_DEVICES", "PS_GPU_MEMORY_LIMIT_MIB", "LD_LIBRARY_PATH")
    }
    if kind in ("conditioning", "native", "rung"):
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif not env.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("missing allocated GPU")
    env.update(
        {k: str(work) for k in ("HOME", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "TORCH_HOME")}
    )
    env.update(
        PYTHONPATH=str(staged),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        DIFFUSERS_OFFLINE="1",
    )
    cmd = [
        "/usr/bin/unshare",
        "--user",
        "--map-root-user",
        "--net",
        str(interpreter),
        "-B",
        "-m",
        "experiments.tennis.common_al028_finish_isolation",
        "worker",
        hashlib.sha256(encoded).hexdigest(),
    ]
    result = subprocess.run(
        cmd,
        input=encoded,
        cwd=staged,
        env=env,
        timeout=seconds,
        close_fds=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (control / "worker.stdout").write_bytes(result.stdout)
    (control / "worker.stderr").write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(
            f"{kind} worker exit {result.returncode}: {result.stderr[-2400:].decode(errors='replace')}"
        )
    publish = work / "result"
    receipt = json.loads(
        (publish / ("finish.json" if kind == "mixed" else f"{kind}.json")).read_text()
    )
    if receipt["isolation"]["code_manifest_sha256"] != source_manifest[
        "manifest_sha256"
    ] or receipt["carrier_sha256"] != worker_config.get(
        "expected_sha256", worker_config.get("carrier_sha256")
    ):
        raise ValueError("worker returned inconsistent identity")
    publish.rename(output)
    return receipt


def worker(expected_hash: str) -> None:
    raw = sys.stdin.buffer.read(500_000)
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("configuration identity mismatch")
    config = json.loads(raw)
    main_runtime = config["kind"] in ("native", "finish", "mixed", "rung")
    env_sha = manifest.SAM_ENV if main_runtime else manifest.GEN_ENV
    runtime = (Path("/local/users/emanuele/pointstream/cache") / env_sha / "tree").resolve(
        strict=True
    )
    if Path(sys.prefix).resolve() != runtime:
        raise ValueError("unexpected generator environment")
    root = Path(config["code"])
    if Path.cwd() != root or not Path(__file__).resolve().is_relative_to(root):
        raise ValueError("worker code origin")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    ns = {key: os.readlink("/proc/self/ns/" + key) for key in ("user", "net")}
    if any(ns[k] == v for k, v in config["namespaces"].items()) or [
        n for _, n in socket.if_nameindex()
    ] != ["lo"]:
        raise ValueError(
            "source-denied worker requires separate user and loopback-only network namespaces"
        )
    output = Path(config["output"])
    if list(output.iterdir()):
        raise ValueError("worker needs empty scratch")
    is_gpu = config["kind"] in ("generation", "finish", "mixed")
    denied = [Path(p) for p in config["denied"]]
    limits = (config["seconds"],) * 2
    resource.setrlimit(resource.RLIMIT_CPU, limits)
    from experiments.tennis import w8_receiver_bootstrap as boot

    # The packed runtime owns its public dependencies. Restrict individual host
    # hardware/loader leaves as in the qualified W8 worker, never the data root.
    system_paths = [Path(p) for p in policy.SYSTEM_GRANTS if is_gpu or "/nvidia" not in p]
    system_paths.extend(Path(p) for p in policy.APPROVED_LEAVES if is_gpu or "/nvidia" not in p)
    listing_only: tuple[Path, ...] = ()
    prewarm = None
    if is_gpu:
        import torch
        from experiments.tennis import w8_clock_profile as clock
        from experiments.tennis.w8_input_job import synthetic_torch_bootstrap

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized before source denial")
        prewarm = synthetic_torch_bootstrap(torch)
        system_paths.extend(clock.PATHS)
        listing_only = (clock.ROOT,)
    allowed = [root, runtime, output, *(Path(p) for p in config["inputs"])]
    allowed.extend(p.resolve(strict=True) for p in system_paths if p.exists())
    if any(
        p == a or p.is_relative_to(a) for p in denied + [Path(config["canary"])] for a in allowed
    ):
        raise ValueError("allowlist exposes protected file")
    if len(os.listdir("/proc/self/task")) != 1 or any(
        int(fd) > 2 and Path("/proc/self/fd", fd).exists() for fd in os.listdir("/proc/self/fd")
    ):
        raise ValueError("unexpected worker thread or inherited descriptor")
    native_exec: tuple[Path, ...] = ()
    if main_runtime:
        from experiments.tennis import w8_receiver_native_bootstrap as native

        binary = Path(config["settings"]["ffmpeg"]).resolve(strict=True)
        loader = Path("/lib64/ld-linux-x86-64.so.2").resolve(strict=True)
        if (
            binary != (runtime / "bin/ffmpeg").resolve(strict=True)
            or code.sha(binary) != manifest.NATIVE
        ):
            raise ValueError("unrecognized native executable")
        native_exec = (binary, loader)
        allowed.extend(native_exec)
        cmd = native.decode_argv(binary, output / "result/base.ivf", output / "result/base.yuv")
        hook = native._audit_policy(cmd)
    else:
        hook = boot._deny_processes
    abi, rights = policy.restrict(
        list(dict.fromkeys(allowed)), output, executables=native_exec, listing_only=listing_only
    )
    sys.addaudithook(hook)
    failures = boot._check_denials(denied)
    if is_gpu and (torch.cuda.is_initialized() or synthetic_torch_bootstrap(torch) != prewarm):
        raise RuntimeError("CPU bootstrap changed across source denial")
    try:
        fd = os.open(config["canary"], os.O_WRONLY)
    except OSError as exc:
        if exc.errno not in (1, 13):
            raise
    else:
        os.close(fd)
        raise RuntimeError("protected canary writable")
    isolation: dict[str, Any] = {
        "landlock_abi": abi,
        "handled_rights": rights,
        "denials": failures,
        "namespaces": ns,
        "interfaces": ["lo"],
        "mutation_denied": True,
        "extra_fds": False,
        "pre_sandbox_threads": 1,
        "cpu_bootstrap_hash": prewarm,
        "listing_only": list(map(str, listing_only)),
        "executables": list(map(str, native_exec)),
        "allowed": list(map(str, allowed)),
        "code_manifest_sha256": config["code_manifest_sha256"],
    }
    options = config["settings"]
    carrier_hash = options.get("expected_sha256", options.get("carrier_sha256"))
    bound = manifest.verify(Path(options["manifest"]), carrier_hash)
    isolation.update(
        manifest_sha256=bound["manifest_sha256"],
        manifest_bytes=bound["manifest_bytes"],
        runtime_archive_sha256=env_sha,
    )
    if main_runtime:
        from experiments.tennis import common_player_al028_finish as finish

        options["output"] = str(output / "result")
        if config["kind"] in ("finish", "mixed"):
            finish.run(options, isolation)
        elif config["kind"] == "rung":
            from experiments.tennis import med_mixed_rung

            med_mixed_rung.run(options, isolation)
        else:
            from experiments.tennis import w8_receiver_finish as w8

            target = Path(options["output"])
            target.mkdir()
            _, base, _ = finish.geometry(Path(options["carrier"]).read_bytes(), carrier_hash)
            ivf, yuv = target / "base.ivf", target / "base.yuv"
            ivf.write_bytes(base)
            if (
                code.sha(binary) != w8.FFMPEG_SHA
                or code.sha(ivf)
                != "8466fe0f097c79a2383351ea9ccce438ea62cd1e0283d25d5c68b08ee4b96c8d"
            ):
                raise ValueError("wrong native decoder or paid base")
            subprocess.run(
                [
                    str(binary),
                    "-nostdin",
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-c:v",
                    "libdav1d",
                    "-i",
                    str(ivf),
                    "-pix_fmt",
                    "yuv420p",
                    "-f",
                    "rawvideo",
                    "-n",
                    str(yuv),
                ],
                check=True,
                capture_output=True,
                timeout=45,
            )
            yuv_sha = code.sha(yuv)
            if yuv_sha != finish.YUV_SHA:
                raise ValueError("source-denied native frame identity mismatch")
            (target / "native.json").write_text(
                json.dumps(
                    {
                        "carrier_sha256": carrier_hash,
                        "isolation": isolation,
                        "native_yuv_sha256": yuv_sha,
                        "native_bytes": yuv.stat().st_size,
                        "frame_count": 240,
                    }
                )
            )
        return
    # The JCV archive's canonical public AnimateAnyone source precedes other
    # site modules. No external package/source checkout is mounted.
    aa = runtime / "opt/Moore-AnimateAnyone"
    sys.path.insert(1, str(aa))
    from experiments.tennis import common_receiver_generate as generate

    generate.run(
        Path(options["carrier"]),
        options["expected_sha256"],
        {k: Path(p) for k, p in options.get("weights", {}).items()},
        Path(options["fit"]) if is_gpu else None,
        output / "result",
        frames=options["frames"],
        units=options["units"],
        isolation=isolation,
        conditioning_only=not is_gpu,
    )


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "worker":
    worker(sys.argv[2])
