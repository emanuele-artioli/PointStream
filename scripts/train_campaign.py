"""Successive-halving driver for the G2 training campaign (report 10, Phase 5.4).

**This script trains for real when invoked without --dry-run.** It is the
"launch variants under successive halving" half of the protocol; the probe
set (`scripts/select_probe_set.py`) and checkpoint scoring
(`scripts/eval_checkpoint.py`) are its building blocks. Per this workstream's
explicit scope gate: this driver is meant to be validated with tiny budgets
(a couple of epochs, a couple of minutes) — a human decides separately when
to point it at the real multi-day campaign with realistic epoch budgets.

Successive-halving rule (exact)
--------------------------------
1. Rung 0: every alive variant trains for `--initial-epochs` (from scratch).
2. After training, each variant's checkpoint is scored on the probe set
   (`scripts/eval_checkpoint.py`'s `evaluate_checkpoint`), producing an
   aggregate PSNR/SSIM/VMAF/FVD(+LPIPS) record.
3. Rank variants by a composite score: for each metric, min-max normalize
   across the variants *present this rung* (lower-is-better metrics — FVD,
   the uncalibrated LPIPS-like distance — are flipped so higher-normalized
   always means better), then average the normalized metrics available for
   that variant. A metric is only used for ranking if at least 2 variants
   reported a non-None value for it that rung.
4. Keep the top `ceil(n_alive / 2)` variants (classic successive halving);
   with only 3 initial candidates this means: rung 0 keeps 2 of 3, rung 1
   keeps 1 of 2, rung 2 is a no-op (1 of 1) and the loop stops.
5. Survivors' budget **doubles** (rung 1 trains to 2x rung 0's cumulative
   epochs, rung 2 to 4x, ...); pix2pix/spade4tennis resume from their own
   checkpoint to an absolute cumulative-epoch target, ControlNet resumes
   from its saved directory for `rung_epochs = target - previous` more
   epochs (see `build_train_command`'s docstring for why the two families
   differ here).
6. **By default this driver runs exactly one rung per invocation and stops**,
   printing the ranking and survivors, so a human reviews the probe-log
   curve before the next rung's (bigger) budget is authorized — re-run with
   the same `--campaign-dir` to continue. `--auto-continue` opts into
   running all remaining rungs unattended; this flag is for a
   human-approved real campaign, not for validating the harness.

Animate-Anyone note (report 10 Phase 5.4(b) asks to "note if a training
script exists"): it does — `animate_anyone.scripts.train_stage_1`/
`train_stage_2` (installed from the vendored `moore-animateanyone` package),
driven by YAML configs under `assets/animate-anyone/configs/` rather than
argparse data-root/epochs flags, and its dataset loader expects a separate
`extract_meta_info.py` pre-pass to build `meta_paths` before training can
start. Wiring it as a fourth `Variant` kind needs a YAML-templating step
(patching `train_bs`/`max_train_steps`/`data.meta_paths` per rung) that is
out of scope for this pass — **not wired into `default_variants()`**; a
future session should add an `AnimateAnyoneVariant` following the same
`build_train_command` contract once that templating exists.

Usage
-----
    conda run -n pointstream python scripts/train_campaign.py \\
        --campaign-dir outputs/campaign/g2_smoke \\
        --manifest assets/probe_set/manifest.json \\
        --data-root assets/probe_set/training_view \\
        --eval-dataset-root assets/dataset \\
        --initial-epochs 1
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from scripts.eval_checkpoint import append_jsonl_log, evaluate_checkpoint, load_manifest

LOWER_IS_BETTER = {"fvd", "lpips_vgg_uncalibrated"}
HIGHER_IS_BETTER = {"psnr_mean", "ssim_mean", "vmaf_mean"}
RANKED_METRICS = tuple(sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER))


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    name: str
    arch: str  # eval_checkpoint.py --arch
    kind: str  # "pix2pix" | "spade4tennis" | "controlnet"
    condition_type: str | None = None  # controlnet only
    model_size: str = "lite"  # spade4tennis only


def default_variants() -> list[Variant]:
    """The 3 wrapped candidates (report 10 Phase 5.4(b)); Animate-Anyone
    deliberately not included yet — see module docstring."""
    return [
        Variant(name="pix2pix", arch="pix2pix", kind="pix2pix"),
        Variant(name="spade4tennis_lite", arch="spade4tennis", kind="spade4tennis", model_size="lite"),
        Variant(name="controlnet_pose", arch="controlnet", kind="controlnet", condition_type="pose"),
    ]


def checkpoint_path_for_eval(variant: Variant, ckpt_dir: Path) -> Path:
    if variant.kind == "pix2pix":
        return ckpt_dir / "pix2pix_generator.pt"
    if variant.kind == "spade4tennis":
        return ckpt_dir / f"spade4tennis_{variant.model_size}_generator.pt"
    if variant.kind == "controlnet":
        return ckpt_dir  # diffusers ControlNetModel.from_pretrained expects a directory
    raise ValueError(f"unknown variant kind: {variant.kind}")


# ---------------------------------------------------------------------------
# Command building (pure — no subprocess execution, so this is unit-testable)
# ---------------------------------------------------------------------------


def build_train_command(
    variant: Variant,
    data_root: Path,
    ckpt_dir: Path,
    cumulative_epochs: int,
    rung_epochs: int,
    resume: bool,
    python_bin: str | None = None,
    batch_size: str | None = None,
) -> list[str]:
    """Builds the training-script invocation for one variant's rung.

    pix2pix/spade4tennis express "how far to train" as an *absolute* epoch
    target plus --resume (their own scripts track how many epochs a
    checkpoint has already seen and pick up from there). ControlNet's script
    has no such numeric resume convention — each invocation trains
    `rung_epochs` *more* epochs on top of whatever `--controlnet-model-id`
    points at (the previous rung's saved directory), or starts
    `--from-scratch` on rung 0.

    `batch_size` (str, since pix2pix/spade4tennis accept "auto") is passed
    through as-is when given; left at each script's own default otherwise.
    Needed for tiny smoke-test datasets — pix2pix/spade4tennis's "auto"
    default (64/32) plus drop_last=True silently empties the dataloader
    when the dataset has fewer images than that.
    """
    py = python_bin or sys.executable

    if variant.kind == "pix2pix":
        cmd = [
            py, "scripts/train_pix2pix.py",
            "--data-root", str(data_root),
            "--epochs", str(cumulative_epochs),
            "--out-weights", str(ckpt_dir / "pix2pix_generator.pt"),
            "--checkpoint-path", str(ckpt_dir / "pix2pix_checkpoint.pt"),
            "--sample-dir", str(ckpt_dir / "samples"),
        ]
        if batch_size is not None:
            cmd.extend(["--batch-size", batch_size])
        if resume:
            cmd.append("--resume")
        return cmd

    if variant.kind == "spade4tennis":
        cmd = [
            py, "scripts/train_spade4tennis.py",
            "--model-size", variant.model_size,
            "--data-root", str(data_root),
            "--epochs", str(cumulative_epochs),
            "--out-weights", str(ckpt_dir / f"spade4tennis_{variant.model_size}_generator.pt"),
            "--checkpoint-path", str(ckpt_dir / f"spade4tennis_{variant.model_size}_checkpoint.pt"),
            "--sample-dir", str(ckpt_dir / "samples"),
        ]
        if batch_size is not None:
            cmd.extend(["--batch-size", batch_size])
        if resume:
            cmd.append("--resume")
        return cmd

    if variant.kind == "controlnet":
        cmd = [
            py, "scripts/train_controlnet.py",
            "--data-root", str(data_root),
            "--condition-type", variant.condition_type or "pose",
            "--epochs", str(rung_epochs),
            "--output-dir", str(ckpt_dir),
        ]
        if resume:
            cmd.extend(["--controlnet-model-id", str(ckpt_dir)])
        else:
            cmd.append("--from-scratch")
        return cmd

    raise ValueError(f"unknown variant kind: {variant.kind}")


# ---------------------------------------------------------------------------
# Non-overlap safety net
# ---------------------------------------------------------------------------


def verify_data_root_excludes_probe_set(data_root: Path, manifest: dict) -> list[str]:
    """Returns the list of probe-set keys that ARE reachable under data_root
    (i.e. leaked into training) — empty means the exclusion held."""
    violations = []
    for key in manifest["excluded_training_keys"]:
        video, scene, track = key.split("/")
        if (data_root / video / "segmentations" / scene / track).exists():
            violations.append(key)
    return violations


# ---------------------------------------------------------------------------
# Ranking / successive halving
# ---------------------------------------------------------------------------


def rank_variants(aggregate_by_variant: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, float]]:
    """Composite-score ranking; see module docstring step 3 for the exact rule."""
    normalized: dict[str, dict[str, float]] = {v: {} for v in aggregate_by_variant}

    for key in RANKED_METRICS:
        values = {v: agg[key] for v, agg in aggregate_by_variant.items() if agg.get(key) is not None}
        if len(values) < 2:
            continue
        lo, hi = min(values.values()), max(values.values())
        span = hi - lo
        for v, val in values.items():
            norm = 0.5 if span == 0 else (val - lo) / span
            if key in LOWER_IS_BETTER:
                norm = 1.0 - norm
            normalized[v][key] = norm

    composite = {v: (sum(scores.values()) / len(scores) if scores else 0.0) for v, scores in normalized.items()}
    ranked = sorted(composite, key=lambda v: (-composite[v], v))
    return ranked, composite


def promote_survivors(ranked: list[str]) -> list[str]:
    if len(ranked) <= 1:
        return ranked
    keep = math.ceil(len(ranked) / 2)
    return ranked[:keep]


# ---------------------------------------------------------------------------
# Campaign state (JSON on disk, so a rung can be re-run/resumed by invocation)
# ---------------------------------------------------------------------------


def init_state(variants: list[Variant]) -> dict[str, Any]:
    return {
        "rung": 0,
        "alive": [v.name for v in variants],
        "cumulative_epochs": {v.name: 0 for v in variants},
        "variants": {v.name: asdict(v) for v in variants},
        "history": [],
    }


def load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def variants_from_state(state: dict[str, Any]) -> dict[str, Variant]:
    return {name: Variant(**fields) for name, fields in state["variants"].items()}


# ---------------------------------------------------------------------------
# Rung execution (subprocess for training; in-process for scoring)
# ---------------------------------------------------------------------------


def halved_batch_size(batch_size: str | None) -> str | None:
    """Halve a numeric --train-batch-size for a retry; leave non-numeric values (e.g. "auto") unchanged."""
    if batch_size is None or not batch_size.isdigit():
        return batch_size
    return str(max(1, int(batch_size) // 2))


def run_training_subprocess(cmd: list[str], log_path: Path, timeout: float | None, repo_root: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        proc = subprocess.run(cmd, cwd=str(repo_root), stdout=log_file, stderr=subprocess.STDOUT, timeout=timeout)
    return proc.returncode


def eval_variant(
    variant: Variant,
    ckpt_dir: Path,
    manifest: dict,
    eval_dataset_root: Path,
    step: int,
    rung: int,
    log_path: Path,
    device: str,
    img_size: int,
    metrics: tuple[str, ...],
    include_lpips: bool,
    eval_steps: int,
) -> dict[str, Any]:
    checkpoint = checkpoint_path_for_eval(variant, ckpt_dir)
    condition_type = variant.condition_type if variant.kind == "controlnet" else None

    arch_kwargs = {"num_inference_steps": eval_steps} if variant.kind == "controlnet" else {}

    result = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        arch=variant.arch,
        manifest=manifest,
        dataset_root=eval_dataset_root,
        img_size=img_size,
        device=device,
        metrics=metrics,
        include_lpips=include_lpips,
        fps=24.0,
        condition_type=condition_type,
        arch_kwargs=arch_kwargs,
    )

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rung": rung,
        "step": step,
        "variant": variant.name,
        "arch": variant.arch,
        "eval_steps": eval_steps,
        "checkpoint": str(checkpoint),
        **result["aggregate"],
    }
    append_jsonl_log(log_path, record)
    return result["aggregate"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Probe-excluded training view (--data-root for the wrapped training scripts)")
    parser.add_argument("--eval-dataset-root", type=Path, default=Path("assets/dataset"), help="Full (unfiltered) dataset root, for reading probe-clip ground truth")
    parser.add_argument("--initial-epochs", type=int, default=2)
    parser.add_argument("--max-rungs", type=int, default=None)
    parser.add_argument("--eval-img-size", type=int, default=512, help="Eval output resolution (square)")
    parser.add_argument("--eval-steps", type=int, default=10, help="Inference steps for diffusion variants during campaign eval")
    parser.add_argument("--eval-metrics", type=str, default="psnr,ssim,vmaf", help="Comma separated metrics")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-timeout-sec", type=float, default=None)
    parser.add_argument("--train-batch-size", type=str, default=None, help="Passed through to pix2pix/spade4tennis --batch-size (e.g. for tiny smoke datasets)")
    parser.add_argument("--auto-continue", action="store_true", help="Run all remaining rungs unattended (NOT for harness validation — see module docstring)")
    parser.add_argument(
        "--prune-on-train-failure",
        action="store_true",
        help="Rank/prune a rung even when a variant's training subprocess failed (old, dangerous "
        "behavior — an infra failure such as CUDA OOM would be scored as a quality loss). Default "
        "off: a training failure that survives one halved-batch-size retry aborts the rung without "
        "ranking or mutating state, so a human can fix the infra issue and resume.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print training commands without executing them or scoring")
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated subset of default_variants() names to run (default: all 3). "
        "Useful for a cheap smoke test (e.g. 'pix2pix,spade4tennis_lite' to skip the heavier ControlNet path).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    violations = verify_data_root_excludes_probe_set(args.data_root, manifest)
    if violations:
        parser.error(
            f"--data-root {args.data_root} leaks {len(violations)} probe-set track(s) into training "
            f"(e.g. {violations[:3]}) — refusing to start. Use scripts/select_probe_set.py's "
            f"--materialize-training-view to build a clean --data-root."
        )

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    metrics = tuple(m.strip().lower() for m in args.eval_metrics.split(",") if m.strip())

    state_path = args.campaign_dir / "campaign_state.json"
    log_path = args.campaign_dir / "probe_log.jsonl"

    if state_path.exists():
        state = load_state(state_path)
    else:
        variants = default_variants()
        if args.variants:
            requested = [name.strip() for name in args.variants.split(",") if name.strip()]
            by_name = {v.name: v for v in variants}
            unknown = [name for name in requested if name not in by_name]
            if unknown:
                parser.error(f"unknown --variants entries {unknown}; choices are {sorted(by_name)}")
            variants = [by_name[name] for name in requested]
        state = init_state(variants)
        save_state(state_path, state)

    variant_by_name = variants_from_state(state)
    repo_root = Path(__file__).resolve().parents[1]

    while True:
        if len(state["alive"]) <= 1:
            print(f"Campaign converged: survivor = {state['alive']}")
            break
        if args.max_rungs is not None and state["rung"] >= args.max_rungs:
            print(f"Reached --max-rungs={args.max_rungs}; stopping with alive={state['alive']}")
            break

        rung = state["rung"]
        # All alive variants share the same cumulative target by construction (see docstring step 5).
        prev_cumulative = state["cumulative_epochs"][state["alive"][0]]
        target_epochs = args.initial_epochs if rung == 0 else prev_cumulative * 2

        aggregate_by_variant: dict[str, dict[str, Any]] = {}
        rung_train_failed = False
        for name in state["alive"]:
            variant = variant_by_name[name]
            ckpt_dir = args.campaign_dir / "checkpoints" / name
            prev = state["cumulative_epochs"][name]
            resume = prev > 0
            rung_epochs = target_epochs - prev

            if rung_epochs <= 0:
                print(f"Rung {rung}: {name} already has {prev} cumulative epoch(s) (target {target_epochs}); skipping training.")
                returncode = 0
            else:
                batch_size = args.train_batch_size
                cmd = build_train_command(
                    variant, args.data_root, ckpt_dir, target_epochs, rung_epochs, resume,
                    batch_size=batch_size,
                )

                if args.dry_run:
                    print(f"[dry-run] rung {rung} {name}: {' '.join(cmd)}")
                    continue

                print(f"Rung {rung}: training {name} to {target_epochs} cumulative epoch(s)...")
                log_path_variant = args.campaign_dir / "logs" / f"{name}_rung{rung}.log"
                returncode = run_training_subprocess(cmd, log_path_variant, args.train_timeout_sec, repo_root)

            if returncode != 0:
                retry_batch_size = halved_batch_size(batch_size)
                print(
                    f"WARNING: {name} training subprocess exited {returncode}; retrying once with "
                    f"--train-batch-size {retry_batch_size} (was {batch_size}) in case this was a "
                    "shared-GPU OOM, not a real quality failure."
                )
                retry_cmd = build_train_command(
                    variant, args.data_root, ckpt_dir, target_epochs, rung_epochs, resume,
                    batch_size=retry_batch_size,
                )
                returncode = run_training_subprocess(
                    retry_cmd, args.campaign_dir / "logs" / f"{name}_rung{rung}_retry.log",
                    args.train_timeout_sec, repo_root,
                )

            if returncode != 0:
                print(f"ERROR: {name} training failed twice this rung (exit {returncode}).")
                if args.prune_on_train_failure:
                    print(f"--prune-on-train-failure set: skipping {name}'s eval and continuing this rung.")
                    rung_train_failed = True
                    continue
                print(
                    f"Aborting rung {rung} without ranking or pruning — a training failure is not a "
                    f"quality result. Fix the failure (see {log_path_variant.with_name(log_path_variant.stem + '_retry.log')}) "
                    "and resume by re-running this command; state is unchanged."
                )
                return 1

            aggregate_by_variant[name] = eval_variant(
                variant, ckpt_dir, manifest, args.eval_dataset_root, target_epochs, rung, log_path,
                device, args.eval_img_size, metrics, not args.skip_lpips, args.eval_steps,
            )
            state["cumulative_epochs"][name] = target_epochs

        if args.dry_run:
            print("[dry-run] stopping before eval/ranking")
            break

        if not aggregate_by_variant:
            print("No variant produced a scorable checkpoint this rung; stopping.")
            break

        if rung_train_failed:
            print(
                "NOTE: this rung's ranking excludes at least one variant that failed training twice "
                "(--prune-on-train-failure was set, so it will be pruned as if it lost on quality)."
            )

        ranked, composite = rank_variants(aggregate_by_variant)
        survivors = promote_survivors(ranked)
        pruned = [v for v in state["alive"] if v not in survivors]

        state["history"].append({
            "rung": rung,
            "target_epochs": target_epochs,
            "aggregate_by_variant": aggregate_by_variant,
            "composite_scores": composite,
            "ranked": ranked,
            "survivors": survivors,
            "pruned": pruned,
        })
        state["alive"] = survivors
        state["rung"] = rung + 1
        save_state(state_path, state)

        print(f"Rung {rung} complete. Ranked (best first): {ranked}")
        print(f"Composite scores: {composite}")
        if pruned:
            print(f"Pruned: {pruned}")
        print(f"Survivors advancing to rung {rung + 1} (target {target_epochs * 2} cumulative epochs): {survivors}")

        if not args.auto_continue:
            print(
                "Stopping after one rung (default — a human should review the probe-log curve "
                f"at {log_path} before authorizing the next rung). Re-run with the same "
                "--campaign-dir to continue, or pass --auto-continue to run unattended."
            )
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
