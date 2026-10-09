"""MANO in numpy, for coding experiments that run many poses on CPU (H2).

The same computation as smplx's ``MANOLayer`` with rotation matrices (no hand mean added: HaMeR and
WiLoR predict absolute joint rotations): shape blend shapes, joints from the shaped template, pose
blend shapes, forward kinematics about each rest joint, linear blend skinning. Joints are the 16
kinematic joints plus five fingertip vertices, in OpenPose order, as HaMeR's and WiLoR's
``mano_wrapper`` return them.

Left hands follow the regressors' convention: the right-hand model is posed and its x axis mirrored
(``M = diag(-1, 1, 1)``) before the translation is added.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

TIPS = (744, 320, 443, 554, 671)  # thumb, index, middle, ring, pinky (smplx ``vertex_ids['mano']``)
MANO_TO_OPENPOSE = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)
MIRROR = np.diag([-1.0, 1.0, 1.0])


def rodrigues(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) to rotation matrices (..., 3, 3)."""
    rotvec = np.asarray(rotvec, float)
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    small = theta < 1e-8
    axis = rotvec / np.where(small, 1.0, theta)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    k = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*rotvec.shape[:-1], 3, 3)
    s = np.sin(theta)[..., None]
    c = np.cos(theta)[..., None]
    eye = np.broadcast_to(np.eye(3), k.shape)
    out = eye + s * k + (1 - c) * (k @ k)
    return np.where(small[..., None], eye, out)


def rotvec_of(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) to axis-angle (..., 3), through a quaternion (Shepperd's method,
    stable near π)."""
    m = np.asarray(matrix, float)
    flat = m.reshape(-1, 3, 3)
    diag = np.stack([flat[:, 0, 0], flat[:, 1, 1], flat[:, 2, 2], np.trace(flat, axis1=1, axis2=2)], -1)
    pick = diag.argmax(-1)
    q = np.zeros((len(flat), 4))  # x, y, z, w
    for case in range(4):
        sel = pick == case
        if not sel.any():
            continue
        r = flat[sel]
        if case == 3:
            w = np.sqrt(1 + diag[sel, 3]) / 2
            q[sel] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / (4 * w), (r[:, 0, 2] - r[:, 2, 0]) / (4 * w),
                               (r[:, 1, 0] - r[:, 0, 1]) / (4 * w), w], -1)
        else:
            i, j, k = case, (case + 1) % 3, (case + 2) % 3
            s = np.sqrt(1 + 2 * r[:, i, i] - diag[sel, 3]) / 2
            out = np.zeros((int(sel.sum()), 4))
            out[:, i] = s
            out[:, j] = (r[:, j, i] + r[:, i, j]) / (4 * s)
            out[:, k] = (r[:, k, i] + r[:, i, k]) / (4 * s)
            out[:, 3] = (r[:, k, j] - r[:, j, k]) / (4 * s)
            q[sel] = out
    q *= np.where(q[:, 3:] < 0, -1.0, 1.0)
    sin = np.linalg.norm(q[:, :3], axis=-1)
    angle = 2 * np.arctan2(sin, q[:, 3])
    scale = np.where(sin < 1e-12, 2.0, angle / np.where(sin < 1e-12, 1.0, sin))
    return (q[:, :3] * scale[:, None]).reshape(*m.shape[:-2], 3)


def unwrap_rotvecs(rotvecs: np.ndarray) -> np.ndarray:
    """Make a sequence (T, ..., 3) of axis-angles continuous: each frame takes, of ``r`` and its
    equivalent ``r - 2π r/|r|``, the one nearer the previous frame."""
    out = np.array(rotvecs, float)
    for t in range(1, len(out)):
        r = out[t]
        theta = np.linalg.norm(r, axis=-1, keepdims=True)
        alt = r - 2 * np.pi * r / np.where(theta < 1e-8, 1.0, theta)
        prev = out[t - 1]
        use = np.linalg.norm(alt - prev, axis=-1) < np.linalg.norm(r - prev, axis=-1)
        out[t] = np.where(use[..., None], alt, r)
    return out


class Mano:
    """One side's MANO model (the plain-numpy pickles of ``tools/models/mano_dechumpy.py``)."""

    def __init__(self, arrays: dict[str, Any]) -> None:
        regressor = arrays["J_regressor"]
        self.j_regressor = np.asarray(regressor.todense() if hasattr(regressor, "todense") else regressor, float)
        self.v_template = np.asarray(arrays["v_template"], float)
        self.shapedirs = np.asarray(arrays["shapedirs"], float)
        self.posedirs = np.asarray(arrays["posedirs"], float).reshape(-1, 135)  # (778*3, 135)
        self.weights = np.asarray(arrays["weights"], float)
        self.parents = np.asarray(arrays["kintree_table"], np.int64)[0].copy()
        self.parents[0] = -1
        self.faces = np.asarray(arrays["f"], np.int32)
        self.hands_mean = np.asarray(arrays.get("hands_mean", np.zeros(45)), float)
        self.hands_components = np.asarray(arrays.get("hands_components", np.eye(45)), float)

    @classmethod
    def load(cls, path: str | Path) -> "Mano":
        with open(path, "rb") as handle:
            return cls(pickle.load(handle, encoding="latin1"))

    def shaped(self, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Shaped template (778, 3) and rest joints (16, 3) for one shape."""
        v = self.v_template + self.shapedirs @ np.asarray(betas, float)
        return v, self.j_regressor @ v

    def forward(self, rotmats: np.ndarray, betas: np.ndarray, transl: np.ndarray | None = None, left: bool = False,
                vertices: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        """Joints (N, 21, 3) in OpenPose order and, if asked, vertices (N, 778, 3).

        ``rotmats`` (N, 16, 3, 3): global orientation then 15 joints; ``betas`` (10,) for the batch.
        """
        rotmats = np.asarray(rotmats, float)
        n = rotmats.shape[0]
        v_shaped, rest = self.shaped(betas)
        feature = (rotmats[:, 1:] - np.eye(3)).reshape(n, 135)
        if vertices:
            v_posed = v_shaped[None] + (feature @ self.posedirs.T).reshape(n, 778, 3)
            idx = np.arange(778)
        else:
            idx = np.asarray(TIPS)
            rows = (idx[:, None] * 3 + np.arange(3)).reshape(-1)
            v_posed = v_shaped[idx][None] + (feature @ self.posedirs[rows].T).reshape(n, len(idx), 3)
        world = np.zeros((n, 16, 4, 4))
        local = np.zeros((n, 16, 4, 4))
        local[:, :, :3, :3] = rotmats
        local[:, :, 3, 3] = 1.0
        rel = rest.copy()
        rel[1:] -= rest[self.parents[1:]]
        local[:, :, :3, 3] = rel
        world[:, 0] = local[:, 0]
        for j in range(1, 16):
            world[:, j] = world[:, self.parents[j]] @ local[:, j]
        joints16 = world[:, :, :3, 3].copy()
        skin = world.copy()
        skin[:, :, :3, 3] -= np.einsum("njab,jb->nja", world[:, :, :3, :3], rest)
        w = self.weights[idx]  # (V, 16)
        blend = np.einsum("vj,njab->nvab", w, skin)
        verts = np.einsum("nvab,nvb->nva", blend[..., :3, :3], v_posed) + blend[..., :3, 3]
        tips = verts if not vertices else verts[:, list(TIPS)]
        joints = np.concatenate([joints16, tips], 1)[:, list(MANO_TO_OPENPOSE)]
        if left:
            joints = joints @ MIRROR
            verts = verts @ MIRROR
        if transl is not None:
            t = np.asarray(transl, float).reshape(-1, 1, 3)
            joints = joints + t
            verts = verts + t
        return joints, (verts if vertices else None)


def change_frame(rotation: np.ndarray, offset: np.ndarray, global_orient: np.ndarray, transl: np.ndarray,
                 rest_root: np.ndarray, left: bool) -> tuple[np.ndarray, np.ndarray]:
    """Express a posed hand in another camera frame, ``x' = rotation x + offset``.

    MANO's global rotation turns about the rest root joint, not the origin, so the translation
    absorbs the difference; a left hand (mirrored right model) turns by the mirrored rotation.
    Returns the new global orientation (3, 3) and translation (3,).
    """
    r = MIRROR @ rotation @ MIRROR if left else rotation
    j0 = MIRROR @ rest_root if left else rest_root
    return r @ global_orient, rotation @ transl + offset + rotation @ j0 - j0
