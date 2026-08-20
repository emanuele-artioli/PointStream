"""Keypoint schemas, and projection between them.

A schema is two things at once: the conditioning signal a generator consumes,
and a per-frame metadata cost. Since payload bytes are the currency every
component is judged in, **schema richness is itself an ablation axis** — 133
whole-body joints cost meaningfully more than 17 body joints, and whether the
extra detail buys back more residual than it costs is an open question.

The canonical internal schema for humans is **COCO-WholeBody-133**, the richest
thing any of our backends produces, with declared projections down to whatever a
given consumer wants. Two consequences worth stating plainly:

**The canonical schema is not what goes on the wire.** The transmitted schema is
derived from what the chosen generator actually consumes. Sending 133 joints to a
ControlNet-OpenPose variant that reads 18 of them is wasted payload, and payload
is what we are trying to minimise.

**Absence is represented, not faked.** Every keypoint carries a present flag, so a
17-joint estimator feeding the 133-joint canonical schema marks the other 116
absent rather than filling them with zeros. A zero-filled joint is
indistinguishable from a joint genuinely detected at the origin, and renders as a
limb pointing at the top-left corner.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

# --------------------------------------------------------------------------
# Groups
# --------------------------------------------------------------------------

GROUP_BODY: Final = "body"
GROUP_FEET: Final = "feet"
GROUP_FACE: Final = "face"
GROUP_HANDS: Final = "hands"

ALL_GROUPS: Final = (GROUP_BODY, GROUP_FEET, GROUP_FACE, GROUP_HANDS)


# --------------------------------------------------------------------------
# Joint name vocabularies
# --------------------------------------------------------------------------

#: COCO-17, what YOLO-pose emits.
COCO_17_JOINTS: Final[tuple[str, ...]] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

#: OpenPose-18, what ControlNet-OpenPose conditioners expect. Same anatomy as
#: COCO-17 plus a synthesised `neck` (midpoint of the shoulders), which is why
#: projection between the two is not a plain subset operation.
OPENPOSE_18_JOINTS: Final[tuple[str, ...]] = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)

#: The six foot joints COCO-WholeBody adds on top of COCO-17.
WHOLEBODY_FOOT_JOINTS: Final[tuple[str, ...]] = (
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)

#: AP-10K, the standard animal-pose layout. Present so an animal domain profile
#: has a schema to name without touching the human path.
AP10K_17_JOINTS: Final[tuple[str, ...]] = (
    "left_eye",
    "right_eye",
    "nose",
    "neck",
    "root_of_tail",
    "left_shoulder",
    "left_elbow",
    "left_front_paw",
    "right_shoulder",
    "right_elbow",
    "right_front_paw",
    "left_hip",
    "left_knee",
    "left_back_paw",
    "right_hip",
    "right_knee",
    "right_back_paw",
)


def _indexed(prefix: str, count: int) -> tuple[str, ...]:
    """Positional names for landmark banks with no meaningful individual names.

    The 68 face and 42 hand landmarks are conventionally identified by index, not
    by anatomy. Naming them `face_00…face_67` is honest about that, and keeps
    name-based projection working uniformly.
    """
    return tuple(f"{prefix}_{index:02d}" for index in range(count))


WHOLEBODY_FACE_JOINTS: Final[tuple[str, ...]] = _indexed("face", 68)
WHOLEBODY_HAND_JOINTS: Final[tuple[str, ...]] = (
    *_indexed("left_hand", 21),
    *_indexed("right_hand", 21),
)


# --------------------------------------------------------------------------
# Skeleton edges, for rendering condition images
# --------------------------------------------------------------------------

COCO_17_EDGES: Final[tuple[tuple[str, str], ...]] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
)

WHOLEBODY_FOOT_EDGES: Final[tuple[tuple[str, str], ...]] = (
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_big_toe"),
    ("left_big_toe", "left_small_toe"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_big_toe"),
    ("right_big_toe", "right_small_toe"),
)


# --------------------------------------------------------------------------
# The schema
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KeypointSchema:
    """A named set of joints, their grouping, and how to draw them.

    Args:
        name: Registry key, e.g. ``coco-wholebody-133``.
        joints: Joint names in index order. Index *is* the wire order.
        groups: Group name to the joints it contains. Groups are the unit of
            schema reduction — dropping ``face`` and ``hands`` is what turns the
            canonical 133 into a 23-joint transmission.
        edges: Pairs of joint names to connect when rendering a skeleton image.
        summary: One line, for listings.
    """

    name: str
    joints: tuple[str, ...]
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    edges: tuple[tuple[str, str], ...] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        if len(set(self.joints)) != len(self.joints):
            duplicates = sorted({name for name in self.joints if self.joints.count(name) > 1})
            raise ValueError(f"KeypointSchema {self.name!r} repeats joints: {duplicates}")
        known = set(self.joints)
        for group, members in self.groups.items():
            unknown = sorted(set(members) - known)
            if unknown:
                raise ValueError(
                    f"KeypointSchema {self.name!r} group {group!r} names joints "
                    f"not in the schema: {unknown}"
                )
        for start, end in self.edges:
            for endpoint in (start, end):
                if endpoint not in known:
                    raise ValueError(
                        f"KeypointSchema {self.name!r} has an edge touching unknown "
                        f"joint {endpoint!r}."
                    )

    def __len__(self) -> int:
        return len(self.joints)

    @property
    def index_of(self) -> Mapping[str, int]:
        """Joint name to wire index."""
        return {name: index for index, name in enumerate(self.joints)}

    def group_of(self, joint: str) -> str | None:
        """Which group `joint` belongs to, if any."""
        for group, members in self.groups.items():
            if joint in members:
                return group
        return None

    def reduced(self, keep: Iterable[str], *, name: str | None = None) -> KeypointSchema:
        """A schema keeping only the named groups.

        This is how transmission cost is varied without touching the estimator:
        ``canonical.reduced(["body"])`` yields the 17-joint wire schema for a
        generator that reads only body joints. Joint order is preserved, so
        indices stay comparable across reductions of the same parent.
        """
        keep_set = set(keep)
        unknown = sorted(keep_set - set(self.groups))
        if unknown:
            raise ValueError(
                f"KeypointSchema {self.name!r} has no group(s) {unknown}. "
                f"Groups: {sorted(self.groups)}."
            )
        kept_joints = tuple(
            joint for joint in self.joints if self.group_of(joint) in keep_set
        )
        kept = set(kept_joints)
        return KeypointSchema(
            name=name or f"{self.name}[{'+'.join(sorted(keep_set))}]",
            joints=kept_joints,
            groups={
                group: tuple(joint for joint in members if joint in kept)
                for group, members in self.groups.items()
                if group in keep_set
            },
            edges=tuple(
                (start, end) for start, end in self.edges if start in kept and end in kept
            ),
            summary=f"{self.summary} (reduced to {', '.join(sorted(keep_set))})".strip(),
        )


# --------------------------------------------------------------------------
# The registered schemas
# --------------------------------------------------------------------------

COCO_17 = KeypointSchema(
    name="coco-17",
    joints=COCO_17_JOINTS,
    groups={GROUP_BODY: COCO_17_JOINTS},
    edges=COCO_17_EDGES,
    summary="COCO body keypoints; what YOLO-pose emits.",
)

OPENPOSE_18 = KeypointSchema(
    name="openpose-18",
    joints=OPENPOSE_18_JOINTS,
    groups={GROUP_BODY: OPENPOSE_18_JOINTS},
    edges=(
        *(
            edge
            for edge in COCO_17_EDGES
            if edge != ("left_shoulder", "right_shoulder")
        ),
        ("neck", "nose"),
        ("neck", "left_shoulder"),
        ("neck", "right_shoulder"),
    ),
    summary="OpenPose body layout; what ControlNet-OpenPose conditioners expect.",
)

_WHOLEBODY_JOINTS: Final[tuple[str, ...]] = (
    *COCO_17_JOINTS,
    *WHOLEBODY_FOOT_JOINTS,
    *WHOLEBODY_FACE_JOINTS,
    *WHOLEBODY_HAND_JOINTS,
)

COCO_WHOLEBODY_133 = KeypointSchema(
    name="coco-wholebody-133",
    joints=_WHOLEBODY_JOINTS,
    groups={
        GROUP_BODY: COCO_17_JOINTS,
        GROUP_FEET: WHOLEBODY_FOOT_JOINTS,
        GROUP_FACE: WHOLEBODY_FACE_JOINTS,
        GROUP_HANDS: WHOLEBODY_HAND_JOINTS,
    },
    edges=(*COCO_17_EDGES, *WHOLEBODY_FOOT_EDGES),
    summary="Canonical human schema: body + feet + face + hands; what DWPose emits.",
)

AP10K_17 = KeypointSchema(
    name="ap10k-17",
    joints=AP10K_17_JOINTS,
    groups={GROUP_BODY: AP10K_17_JOINTS},
    edges=(
        ("left_eye", "right_eye"),
        ("nose", "neck"),
        ("neck", "root_of_tail"),
        ("neck", "left_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_front_paw"),
        ("neck", "right_shoulder"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_front_paw"),
        ("root_of_tail", "left_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_back_paw"),
        ("root_of_tail", "right_hip"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_back_paw"),
    ),
    summary="AP-10K animal pose layout.",
)

#: Every schema, by name.
SCHEMAS: Final[Mapping[str, KeypointSchema]] = {
    schema.name: schema
    for schema in (COCO_17, OPENPOSE_18, COCO_WHOLEBODY_133, AP10K_17)
}

#: The canonical internal schema for human domains.
CANONICAL_HUMAN: Final = COCO_WHOLEBODY_133


def schema(name: str) -> KeypointSchema:
    """Look up a schema by name."""
    try:
        return SCHEMAS[name]
    except KeyError:
        raise ValueError(
            f"Unknown keypoint schema {name!r}. Known: {', '.join(sorted(SCHEMAS))}."
        ) from None


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------

#: Joints a target schema defines that no source schema provides directly, and
#: how to synthesise them. OpenPose's `neck` is the only such case today.
DERIVED_JOINTS: Final[Mapping[str, tuple[str, ...]]] = {
    "neck": ("left_shoulder", "right_shoulder"),
}


@dataclass(frozen=True)
class Projection:
    """How to rewrite keypoints from one schema into another.

    Args:
        source: Schema the values are in.
        target: Schema they are wanted in.
        direct: Target index to source index, for joints present in both.
        derived: Target index to the source indices it is the midpoint of.
        absent: Target indices nothing in the source can supply. These are
            marked not-present rather than zero-filled.
    """

    source: KeypointSchema
    target: KeypointSchema
    direct: Mapping[int, int]
    derived: Mapping[int, tuple[int, ...]]
    absent: tuple[int, ...]

    @property
    def is_lossless(self) -> bool:
        """Whether every target joint can be supplied from the source."""
        return not self.absent

    @property
    def coverage(self) -> float:
        """Fraction of target joints the source can supply, derived included."""
        if not len(self.target):
            return 1.0
        return (len(self.direct) + len(self.derived)) / len(self.target)


def project(source: KeypointSchema, target: KeypointSchema) -> Projection:
    """Work out how to map `source` keypoints into `target`.

    Matching is by joint name, which makes projection independent of index
    order and safe to compute between any two schemas. Joints the target wants
    and the source lacks are reported in `absent` — callers mark those
    not-present. Joints the source has and the target does not are simply
    dropped, which is the intended behaviour when reducing a rich estimator's
    output to a lean wire schema.
    """
    source_index = source.index_of
    direct: dict[int, int] = {}
    derived: dict[int, tuple[int, ...]] = {}
    absent: list[int] = []

    for target_idx, joint in enumerate(target.joints):
        if joint in source_index:
            direct[target_idx] = source_index[joint]
            continue
        parents = DERIVED_JOINTS.get(joint)
        if parents and all(parent in source_index for parent in parents):
            derived[target_idx] = tuple(source_index[parent] for parent in parents)
            continue
        absent.append(target_idx)

    return Projection(
        source=source,
        target=target,
        direct=direct,
        derived=derived,
        absent=tuple(absent),
    )


def wire_cost(schema_: KeypointSchema, *, values_per_joint: int = 3) -> int:
    """Values transmitted per frame for one object under `schema_`.

    Default is three — x, y, confidence. This is the number the schema-richness
    ablation trades against reconstruction quality, so it is worth being able to
    state exactly rather than estimating.
    """
    return len(schema_) * values_per_joint


def describe_schemas(names: Sequence[str] | None = None) -> str:
    """A readable table of the registered schemas and their wire cost."""
    chosen = [schema(name) for name in names] if names else list(SCHEMAS.values())
    width = max(len(item.name) for item in chosen)
    lines = ["keypoint schemas:"]
    for item in chosen:
        groups = ", ".join(sorted(item.groups)) or "-"
        lines.append(
            f"  {item.name.ljust(width)}  {len(item):>3} joints, "
            f"{wire_cost(item):>4} values/frame  [{groups}]  {item.summary}"
        )
    return "\n".join(lines)
