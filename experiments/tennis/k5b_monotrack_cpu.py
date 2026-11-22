"""CPU-only adaptation of the fixed zoo inference path; weights and logic unchanged."""

import sys
from pathlib import Path
from typing import Any
import numpy as np
from experiments.tennis.k5b_trackers import ZOO, SIZE, BATCH, Prediction, load_yaml, AttrDict


class CpuZoo:
    """WASB-SBDT's tennis models, imported from its code tree, with its preprocessing and postprocessors."""

    def __init__(self, code: Path, weights: Path, names: tuple[str, ...] = tuple(ZOO)):
        import torch

        root = code / "src"
        sys.path.insert(0, str(root))
        import importlib.util

        import models
        from trackers import build_tracker
        from utils.image import get_affine_transform

        def by_path(name: str) -> Any:
            # detectors/__init__ imports hydra through the detector class, which this wrapper does not use.
            spec = importlib.util.spec_from_file_location(
                f"wasb_{name}", root / "detectors" / f"{name}.py"
            )
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        DeepBallPostprocessor = by_path("deepball_postprocessor").DeepBallPostprocessor
        TracknetV2Postprocessor = by_path("postprocessor").TracknetV2Postprocessor

        if not str(Path(models.__file__).resolve()).startswith(str(root.resolve())):
            raise RuntimeError(f"'models' resolved to {models.__file__}, not WASB's tree")
        self.get_affine_transform = get_affine_transform
        self.build_tracker = build_tracker
        self.settings: dict[str, dict[str, Any]] = {}
        dataloader = load_yaml(root / "configs/dataloader/default.yaml")
        loaded: dict[str, Any] = {}
        for name, s in ((n, ZOO[n]) for n in names):
            model_cfg = load_yaml(root / f"configs/model/{s['model']}.yaml")
            detector_cfg = load_yaml(
                root / f"configs/detector/{s.get('detector', 'tracknetv2')}.yaml"
            )
            detector_cfg["postprocessor"].setdefault("use_hm_weight", s.get("hm_weight", False))
            if "hm_weight" in s:
                detector_cfg["postprocessor"]["use_hm_weight"] = s["hm_weight"]
            detector_cfg["step"] = s["step"]
            cfg = AttrDict.of(
                {
                    "model": model_cfg,
                    "detector": detector_cfg,
                    "dataloader": dataloader,
                    "tracker": load_yaml(root / f"configs/tracker/{s['tracker']}.yaml"),
                    "runner": {"device": "cpu", "gpus": [0]},
                }
            )
            key = s["weights"]
            if key not in loaded:
                net = models.build_model(cfg)
                state = torch.load(weights / key, map_location="cpu", weights_only=True)
                net.load_state_dict(state["model_state_dict"])
                loaded[key] = net.to("cpu").eval()
            post = (
                DeepBallPostprocessor
                if detector_cfg["postprocessor"]["name"] == "deepball"
                else TracknetV2Postprocessor
            )(cfg)
            self.settings[name] = {"cfg": cfg, "net": loaded[key], "post": post}

    def transform(self, wh: tuple[int, int], inv: int) -> np.ndarray:
        w, h = SIZE
        return self.get_affine_transform(
            np.array([w / 2.0, h / 2.0], np.float32), max(h, w) * 1.0, 0, list(wh), inv=inv
        )

    def run(self, name: str, frames: np.ndarray) -> list[Prediction]:
        """Per frame (visible, x, y) as WASB's eval runner produces it: windows of frames_in at the setting's step,
        each window's outputs on its last frames_out frames (a later window overwrites), then the tracker in order.
        A final window aligned to the clip's end covers frames the step leaves out."""
        import cv2
        import torch

        st = self.settings[name]
        m = st["cfg"]["model"]
        fin, fout, step = m["frames_in"], m["frames_out"], st["cfg"]["detector"]["step"]
        inp = (m["inp_width"], m["inp_height"])
        trans_in = self.transform(inp, 0)
        trans_out_inv = torch.from_numpy(
            np.stack([self.transform((m["out_width"], m["out_height"]), 1)])
        ).float()
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        n = len(frames)
        starts = list(range(0, n - fin + 1, step))
        if starts and starts[-1] != n - fin:
            starts.append(n - fin)
        cache: dict[int, np.ndarray] = {}

        def prep(t: int) -> np.ndarray:
            if t not in cache:
                img = (
                    cv2.warpAffine(
                        np.asarray(frames[t]), trans_in, inp, flags=cv2.INTER_LINEAR
                    ).astype(np.float32)
                    / 255.0
                )
                cache[t] = np.moveaxis((img - mean) / std, -1, 0)
            return cache[t]

        dets: dict[int, list[dict[str, Any]]] = {}
        with torch.no_grad():
            for b in range(0, len(starts), BATCH):
                batch = []
                for s in starts[b : b + BATCH]:
                    x = [prep(t) for t in range(s, s + fin)]
                    if m.get("rgb_diff"):
                        x[0] = np.abs(x[1] - x[0])
                    batch.append(np.concatenate(x, 0))
                for t in [t for t in cache if t < starts[min(b + BATCH, len(starts)) - 1]]:
                    del cache[t]
                preds = st["net"](torch.from_numpy(np.stack(batch)).to("cpu"))
                res = st["post"].run(preds, {0: trans_out_inv.repeat(len(batch), 1, 1)})
                for i, s in enumerate(starts[b : b + BATCH]):
                    for j in range(fout):
                        t = s + fin - fout + j
                        dets[t] = [
                            {"xy": np.asarray(xy), "score": float(sc)}
                            for xy, sc in zip(res[i][j][0]["xys"], res[i][j][0]["scores"])
                        ]
        tracker = self.build_tracker(st["cfg"])
        tracker.refresh()
        out: list[Prediction] = []
        for t in range(n):
            r = tracker.update(dets.get(t, []))
            out.append(
                (
                    bool(r["visi"]),
                    float(r["x"]) if r["visi"] else 0.0,
                    float(r["y"]) if r["visi"] else 0.0,
                )
            )
        return out
