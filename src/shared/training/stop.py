"""Stop a training run that cannot clear the coding-task bar.

The pose-ref retrain burned ~14 GPU hours on a series that was flat from
epoch 1 because nothing stopped it. Diffusion noise-prediction loss fell the
whole time; sample quality did not. This module never sees that loss.

The criterion is the real task: region-scoped PSNR and calibrated LPIPS on the
coding task (appearance from a keyframe, conditioning from a later frame),
against the static-copy floor measured on the same items.

Nothing this module records is citable. It is a stopping signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

# Roster floors from plans/done/RESEARCH-HISTORY.md §2.10, 12 probe clips, offsets 1–8.
# A stop-eval on a *subset* must replace these with a measured floor written
# to bounds.json before training starts. These are the fallback labels only.
DEFAULT_FLOOR_PSNR = 13.51
DEFAULT_FLOOR_LPIPS = 0.4505
DEFAULT_NULL_LPIPS = 0.7358

MIN_EPOCHS = 3
PATIENCE = 3
FLAT_EPS_LPIPS = 0.005


@dataclass(frozen=True)
class StopBounds:
    """Bars written to disk before the first training step."""

    floor_psnr: float
    floor_lpips: float
    null_lpips: float = DEFAULT_NULL_LPIPS
    min_epochs: int = MIN_EPOCHS
    patience: int = PATIENCE
    flat_eps_lpips: float = FLAT_EPS_LPIPS
    source: str = ""

    def __post_init__(self) -> None:
        if self.min_epochs < 1:
            raise ValueError(f"min_epochs must be >= 1, got {self.min_epochs}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.floor_lpips >= self.null_lpips:
            raise ValueError(
                "static-copy LPIPS must sit below the unrelated-image null; "
                f"got floor={self.floor_lpips} null={self.null_lpips}"
            )


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str
    keep_as_best: bool
    epoch: int
    lpips: float
    psnr: float


@dataclass
class _Eval:
    epoch: int
    step: int | None
    lpips: float
    psnr: float


def write_bounds(path: Path, bounds: StopBounds, extra: dict[str, Any] | None = None) -> None:
    """Write the bar before any training step. Call this first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "written_before_training": True,
        "task": "coding: appearance from keyframe, condition from a later frame",
        "ranking_key": "lpips",
        "ranking_key_direction": "lower_better",
        "psnr_scope": "object mask",
        "lpips_scope": "bounding box of the object mask (LPIPS is a patch metric)",
        "instrument_lpips_range": (
            f"identical 0; static-copy floor {bounds.floor_lpips}; "
            f"unrelated-image null {bounds.null_lpips}"
        ),
        "instrument_psnr_range": (
            f"identical +inf; static-copy floor {bounds.floor_psnr} dB; "
            "typical coded object 8–16 dB on this task"
        ),
        "do_not_stop_on_diffusion_loss": True,
        "not_citable": True,
        **asdict(bounds),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2) + "\n")


class TaskStopRule:
    """Per-epoch (and optional mid-epoch) observer. Never stops on loss."""

    def __init__(self, bounds: StopBounds, output_dir: Path) -> None:
        self.bounds = bounds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[_Eval] = []
        self.best_lpips: float | None = None
        self.best_epoch: int | None = None
        self._no_improve_epochs: int = 0
        self.stopped: bool = False
        self.stop_reason: str = ""
        write_bounds(self.output_dir / "bounds.json", bounds)

    def observe(
        self,
        *,
        epoch: int,
        lpips: float,
        psnr: float,
        step: int | None = None,
        kind: str = "epoch",
    ) -> StopDecision:
        """Record one coding-task eval. ``kind='mid'`` cannot stop before ``min_epochs``."""
        if kind not in {"epoch", "mid"}:
            raise ValueError(f"kind must be 'epoch' or 'mid', got {kind!r}")
        if self.stopped:
            return StopDecision(
                stop=True,
                reason=self.stop_reason,
                keep_as_best=False,
                epoch=epoch,
                lpips=lpips,
                psnr=psnr,
            )

        row = _Eval(epoch=epoch, step=step, lpips=float(lpips), psnr=float(psnr))
        self.history.append(row)

        keep_as_best = self.best_lpips is None or row.lpips < self.best_lpips
        if keep_as_best:
            self.best_lpips = row.lpips
            self.best_epoch = epoch
            if kind == "epoch":
                self._no_improve_epochs = 0
        elif kind == "epoch":
            self._no_improve_epochs += 1

        reason = ""
        stop = False
        if epoch >= self.bounds.min_epochs:
            if kind == "epoch" and self._no_improve_epochs >= self.bounds.patience:
                stop = True
                reason = (
                    f"patience: no LPIPS improvement for {self._no_improve_epochs} "
                    f"epochs (best {self.best_lpips:.4f} at epoch {self.best_epoch})"
                )
            elif self._flat_to_down_below_floor():
                stop = True
                reason = (
                    "flat-to-down over the last 3 epoch evals and still below the "
                    f"static-copy floor (LPIPS {self.bounds.floor_lpips:.4f} / "
                    f"PSNR {self.bounds.floor_psnr:.2f} dB)"
                )

        if stop:
            self.stopped = True
            self.stop_reason = reason

        decision = StopDecision(
            stop=stop,
            reason=reason,
            keep_as_best=keep_as_best,
            epoch=epoch,
            lpips=row.lpips,
            psnr=row.psnr,
        )
        self._write_series(decision)
        return decision

    def _epoch_evals(self) -> list[_Eval]:
        """Last observation per epoch number, in epoch order."""
        latest: dict[int, _Eval] = {}
        for row in self.history:
            latest[row.epoch] = row
        return [latest[epoch] for epoch in sorted(latest)]

    def _flat_to_down_below_floor(self) -> bool:
        """Last three *epoch* evals: still worse than a paste, and not improving.

        A slow starter that is actually moving (last LPIPS below first by more
        than ``flat_eps_lpips``) is left alone. Noise around a flat line is not
        a slow start.
        """
        epochs = self._epoch_evals()
        if len(epochs) < 3:
            return False
        last3 = epochs[-3:]
        lpips = [row.lpips for row in last3]
        below_floor = all(value >= self.bounds.floor_lpips for value in lpips)
        net_change = lpips[-1] - lpips[0]
        not_improving = net_change >= -self.bounds.flat_eps_lpips
        return below_floor and not_improving

    def _write_series(self, decision: StopDecision) -> None:
        payload = {
            "not_citable": True,
            "floor_lpips": self.bounds.floor_lpips,
            "floor_psnr": self.bounds.floor_psnr,
            "null_lpips": self.bounds.null_lpips,
            "best_lpips": self.best_lpips,
            "best_epoch": self.best_epoch,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "last_decision": {
                "stop": decision.stop,
                "reason": decision.reason,
                "keep_as_best": decision.keep_as_best,
                "epoch": decision.epoch,
                "lpips": decision.lpips,
                "psnr": decision.psnr,
            },
            "series": [
                {
                    "epoch": row.epoch,
                    "step": row.step,
                    "lpips": row.lpips,
                    "psnr": row.psnr,
                    "beats_floor_lpips": row.lpips < self.bounds.floor_lpips,
                    "beats_floor_psnr": row.psnr > self.bounds.floor_psnr,
                }
                for row in self.history
            ],
        }
        (self.output_dir / "stop_series.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
