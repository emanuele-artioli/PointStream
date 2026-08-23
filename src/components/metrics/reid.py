"""Identity similarity: did the **right body** appear, not just a different one.

Every other metric here measures *distance to the target frame*. That is the
wrong question for one specific job, and `BP12` proved it expensively: the
cross-appearance control was meant to say whether a generator uses the
appearance it is given, and a pasted keyframe — no network at all — topped its
scale, because pasting maximises how much of the reference survives into the
output (``PLAN.md`` §2.10).

This asks a different question. A person re-identification embedding maps a
**full-body crop** into a space where the same person in a different pose,
scale and viewpoint is close, and a different person is far. Scored against the
ground-truth target frame it is *pose-invariant by design*, so it separates
three cases that no distortion metric separates on its own:

| arm | LPIPS to target | this metric | reading |
|---|---|---|---|
| static copy | poor | **high** | right person, wrong pose — which is exactly what a paste is |
| a generator drawing someone else | middling | **low** | the failure that had no name |
| a working generator | good | high | the target |

**A paste scoring high here is correct, not a defect.** Its failure shows up on
the other axis. The pair is the instrument; neither number alone is.

**Not faces.** CSIM / ArcFace is the usual literature answer and does not apply:
this project reconstructs bodies in motion, a player box averages ~88k px in a
4K frame, and the face inside it is a few tens of pixels, often turned away.
A full-body embedding needs no face and generalises past people.

**Calibrate before ranking anything.** ReID backbones are trained on
surveillance footage; broadcast sports crops are out of domain, and an
out-of-domain embedding can be perfectly ordered and still uninterpretable —
which is precisely how the old LPIPS shipped. The anchors, and the
same-player / different-player-same-match separation that decides whether this
metric may be used at all, live in
``tests/invariants/test_metric_calibration.py``.

Higher is better; 1.0 is the same crop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.components.metrics.frames import to_clip

#: OSNet is trained at 256x128 (H x W). Feeding it a square canvas measures a
#: resize as much as a person, so every crop is fitted to this shape.
INPUT_HEIGHT = 256
INPUT_WIDTH = 128

#: ImageNet statistics, which is what the vendored backbone was trained under.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

DEFAULT_CHECKPOINT = Path("assets") / "weights" / "reid" / "osnet_x1_0_msmt17.pth"

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def repo_root() -> Path:
    """The worktree root (the directory that contains ``src/``)."""
    return Path(__file__).resolve().parents[3]


def resolve_checkpoint(checkpoint: str | Path | None = None) -> Path:
    """Absolute path to the ReID weights, or a message saying how to get them.

    Deliberately does not reach for the detection weight resolver: this metric
    must not depend on another axis, and nothing here downloads at runtime.
    """
    path = Path(checkpoint) if checkpoint is not None else repo_root() / DEFAULT_CHECKPOINT
    if not path.is_absolute():
        path = repo_root() / path
    if not path.is_file():
        raise FileNotFoundError(
            f"ReID weights not found at {path}. Fetch osnet_x1_0 trained on "
            "MSMT17 from https://huggingface.co/kaiyangzhou/osnet (MIT, checked "
            "2026-08-23) and save it there. Nothing downloads at runtime."
        )
    return path


def _load_model(checkpoint: Path, device: str) -> Any:
    key = (str(checkpoint), device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import torch

    from src.components.metrics._osnet import osnet_x1_0

    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("state_dict", state)
    state = {key_.replace("module.", "", 1): value for key_, value in state.items()}
    # The classifier head is sized by the training set's identity count and is
    # never used here — eval-mode forward returns the embedding before it.
    identities = int(state["classifier.weight"].shape[0])
    model = osnet_x1_0(num_classes=identities)
    model.load_state_dict(state, strict=True)
    model = model.eval().to(torch.device(device))
    _MODEL_CACHE[key] = model
    return model


class ReidMetric:
    """Cosine similarity between person-ReID embeddings. Higher is better.

    ``extractor`` replaces the network for tests. ``checkpoint`` and ``device``
    are recorded by the caller alongside the score, because an embedding metric
    without its checkpoint is not reproducible.
    """

    name = "reid"

    def __init__(
        self,
        *,
        device: str = "cpu",
        checkpoint: str | Path | None = None,
        extractor: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> None:
        self._device = device
        self._checkpoint = checkpoint
        self._extractor = extractor

    def score(self, reference: np.ndarray, predicted: np.ndarray) -> float:
        """Mean per-frame cosine similarity over a ``(T, H, W, C)`` pair.

        **The two sides need not share a spatial shape**, which is the one place
        this metric departs from every other backend here. Each crop is fitted
        to the backbone's input independently, and being invariant to scale and
        framing is the entire reason for using it — demanding pixel alignment
        would re-impose exactly the constraint it exists to escape. The frame
        counts must still match, because frame *i* is compared with frame *i*.
        """
        ref = to_clip(reference)
        pred = to_clip(predicted)
        if ref.shape[0] != pred.shape[0]:
            raise ValueError(
                "reference and predicted clips must have the same number of "
                f"frames; got {ref.shape[0]} vs {pred.shape[0]}. Spatial shapes "
                "may differ — this metric resizes each crop independently."
            )
        if self._extractor is not None:
            return float(self._extractor(ref, pred))
        return self._score_frames(ref, pred)

    def embed(self, clip: np.ndarray) -> np.ndarray:
        """``(T, 512)`` L2-normalised embeddings. Exposed for calibration."""
        import torch

        model = _load_model(resolve_checkpoint(self._checkpoint), self._device)
        device = torch.device(self._device)
        batch = _to_batch(clip)
        with torch.no_grad():
            features = model(torch.from_numpy(batch).to(device))
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        return np.asarray(features.cpu().numpy(), dtype=np.float64)

    def _score_frames(self, ref: np.ndarray, pred: np.ndarray) -> float:
        left = self.embed(ref)
        right = self.embed(pred)
        return float(np.sum(left * right, axis=1).mean())


def _to_batch(clip: np.ndarray) -> np.ndarray:
    """``(T, H, W, C)`` uint8 -> normalised ``(T, 3, 256, 128)`` float32."""
    import cv2

    frames = to_clip(clip)
    resized = np.stack(
        [
            cv2.resize(
                frame[..., :3], (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR
            )
            for frame in frames
        ]
    ).astype(np.float32) / 255.0
    normalised = (resized - np.asarray(_MEAN, dtype=np.float32)) / np.asarray(
        _STD, dtype=np.float32
    )
    return np.ascontiguousarray(np.transpose(normalised, (0, 3, 1, 2)))


def build(**kwargs: Any) -> ReidMetric:
    return ReidMetric(**kwargs)
