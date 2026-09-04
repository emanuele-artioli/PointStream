"""The background as a stream across scenes, not a still re-sent per scene.

`plans/done/BP24-findings.md` §18 measured that coding the next plate as a P-frame
against a previous one saves 31-53% with av1, and §19 showed the saving is
**causal**: unchanged to the byte with `-lag-in-frames 0`. `plans/done/BP30-background-stream.md`
is the design for spending that saving. This module is the transmitter.

**The one thing that can go wrong quietly, and the property that stops it.**
A P-frame predicts from the *reconstructed* reference, never the original. If
the encoder predicts from pixels the client does not hold, the two drift apart
and every later scene is decoded against the wrong picture -- silently, because
a drifted reconstruction is still an image. The paper already commits to this
discipline one level down (`sections/system_design.tex`: the residual is
computed against the codec-decoded background). The same rule applies here.

The way this module keeps both sides identical is to never ask them to agree
about pixels at all. **A scene's payload is the bytes of one frame in a
low-delay elementary stream, and the client reconstructs by decoding the
payloads along that frame's chain.** Encoder and client therefore run the same
decoder over the same bytes, so their reconstructions are equal by construction
rather than by careful arithmetic.

That works only because a low-delay encode is **prefix-stable**: the bytes of
frame *i* do not change when frame *i+1* is appended. Measured on this host
(ffmpeg n7.1.1, libaom-av1, `-usage realtime -lag-in-frames 0 -bf 0`): the
packets for a 2- and 3-frame prefix are byte-identical to the first 2 and 3
packets of the 4-frame encode. That is the same fact §19 established from the
other direction, and it is what makes payload *n* final the moment it is
emitted. `_assert_prefix_stable` re-checks it on every real encode rather than
trusting it, because a flag existing is not a feature working.

**Chains, not one flat stream.** Which earlier reconstruction a scene predicts
from is an axis (brief §3), and ffmpeg's CLI cannot signal an arbitrary
long-term reference. So each scene records the *path* of scenes from its root
keyframe down to itself. Encoding that path reproduces the reference exactly;
the client, which kept the payloads it received, decodes the same path. Under
`last` the path is the whole prefix and this degenerates to the ordinary
growing stream the brief calls equivalent.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from functools import lru_cache
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from src.components.codec import tools

#: Reference-selection modes, `plans/done/BP30-background-stream.md` §3.
REFERENCE_FIRST: Final = "first"
REFERENCE_LAST: Final = "last"
REFERENCE_BEST_SCORED: Final = "best-scored"
REFERENCE_PERIODIC_I: Final = "periodic-i"

REFERENCE_MODES: Final[tuple[str, ...]] = (
    REFERENCE_FIRST,
    REFERENCE_LAST,
    REFERENCE_BEST_SCORED,
    REFERENCE_PERIODIC_I,
)

#: `never` as a keyframe interval means "one keyframe, at the start".
KEYFRAME_NEVER: Final = 0


def context_reset_indices(context_ids: Sequence[str]) -> tuple[int, ...]:
    """Scene indices that start a new independently coded background.

    The **continuous** AV1/VVC control must reset at exactly these indices —
    the same boundaries where PointStream starts a new background context —
    and nowhere else. The **segmented** control resets every scene, which is
    ``tuple(range(n))``, and is a different product.
    """
    if not context_ids:
        return ()
    resets = [0]
    previous = context_ids[0]
    for index, current in enumerate(context_ids[1:], start=1):
        if current != previous:
            resets.append(index)
            previous = current
    return tuple(resets)


def segmented_reset_indices(n_scenes: int) -> tuple[int, ...]:
    """Every scene is an independent intra. Not the continuous control."""
    if n_scenes < 0:
        raise ValueError(f"n_scenes must be >= 0, got {n_scenes}")
    return tuple(range(n_scenes))


def scene_groups(context_ids: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """Half-open ``[start, end)`` ranges of consecutive scenes in one context.

    The continuous AV1/VVC control encodes each range as one sequence.
    The segmented control encodes ``(i, i+1)`` for every scene. Both lists
    of start indices must match ``context_reset_indices`` /
    ``segmented_reset_indices`` of the PointStream configuration they
    compare against — otherwise the reference is resetting more often, or
    less often, than the system under test.
    """
    resets = context_reset_indices(context_ids)
    if not resets:
        return ()
    ends = (*resets[1:], len(context_ids))
    return tuple((int(start), int(end)) for start, end in zip(resets, ends, strict=True))


@dataclass(frozen=True)
class StreamCodec:
    """One encoder, pinned to a low-delay configuration and a raw container.

    The container matters as much as the flags. A payload has to be a run of
    bytes the client can concatenate with the payloads before it and hand to a
    decoder, so the elementary stream (OBU for AV1, Annex-B for AVC/HEVC) is
    the format -- not mkv, whose per-frame framing lives in the container.

    Args:
        name: Short arm name used in results.
        encoder: ffmpeg encoder to invoke.
        container: ffmpeg muxer/demuxer for the elementary stream.
        low_delay: Flags that forbid the encoder from looking ahead. Findings
            §19: `rc-lookahead=0` is *not* one of these for x265 -- it is the
            rate-control lookahead, and zeroing it made a P-frame come back
            larger than a fresh intra.
    """

    name: str
    encoder: str
    container: str
    low_delay: tuple[str, ...]


#: av1 is the arm that matters: findings §19 measured its saving unchanged to
#: the byte under low delay, because `-usage realtime` was already
#: lookahead-free. x265 is kept as a contrast, not as an equal -- it saves 12%
#: on one pair and loses 6% on the other, so the saving is a property of av1's
#: inter tools rather than of inter coding.
CODECS: Final[dict[str, StreamCodec]] = {
    "av1": StreamCodec(
        name="av1",
        encoder="libaom-av1",
        container="obu",
        low_delay=("-cpu-used", "8", "-usage", "realtime", "-lag-in-frames", "0", "-bf", "0"),
    ),
    "hevc": StreamCodec(
        name="hevc",
        encoder="libx265",
        container="hevc",
        low_delay=("-preset", "veryfast", "-bf", "0", "-x265-params", "log-level=none:bframes=0"),
    ),
    "avc": StreamCodec(
        name="avc",
        encoder="libx264",
        container="h264",
        low_delay=("-preset", "veryfast", "-bf", "0", "-x264-params", "log-level=none:bframes=0"),
    ),
}


@lru_cache(maxsize=1)
def _ffmpeg() -> tuple[str, str]:
    """The ffmpeg binary and its ffprobe sibling, resolved once.

    `resolve_ffmpeg` shells out to `ffmpeg --version` and a feature probe every
    time it is called. A sweep runs hundreds of encodes, and on this NFS home
    that overhead is minutes of nothing. Resolved once and cached; the path and
    version still reach the results, so the run still records which binary it
    actually ran (`AGENTS.md`: resolve external tools by path and version).
    """
    tool = tools.resolve_ffmpeg()
    return tool.path, str(Path(tool.path).with_name("ffprobe"))


def ffmpeg_provenance() -> dict[str, str]:
    """Path and self-reported version of the binary this module encodes with."""
    tool = tools.resolve_ffmpeg()
    return {"path": tool.path, "version": tool.version}


class StreamDrift(RuntimeError):
    """Encoder and client would not hold the same reconstruction."""


def canny_edges(image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """Binary edge map, the structure a codec actually spends bits on.

    Findings §18 measured the pair *further apart* in PSNR (federer, 15.10 dB)
    saving more than the closer one (alcaraz, 13.75 dB), so pixel distance does
    not predict coding distance. What survives motion compensation is residual
    *structure* -- edges that do not line up -- which is why the reference score
    is an edge comparison rather than an MSE.
    """
    import cv2

    array = np.asarray(image, dtype=np.uint8)
    grey = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY) if array.ndim == 3 else array
    return cv2.Canny(grey, low, high) > 0


def canny_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two edge maps. Higher is a better reference.

    Returns 1.0 for two edgeless images, which is the honest answer: neither
    carries structure to mispredict.
    """
    edges_a, edges_b = canny_edges(a), canny_edges(b)
    union = int(np.count_nonzero(edges_a | edges_b))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(edges_a & edges_b)) / float(union)


@dataclass(frozen=True)
class ScenePayload:
    """One scene's transmission: the bytes, and how to decode them.

    Args:
        index: Position of this scene in the sequence.
        chain: Scene indices whose payloads precede this one, this scene last.
            A single-element chain is a keyframe.
        payload: The scene's own bytes. This is the marginal wire cost.
        picture_type: `I`, `P`, or `B`. A `B` here means the encode was not
            causal whatever the flags claimed, and is refused rather than
            reported.
        reference: Scene this one predicts from, `None` for a keyframe.
        mode: Reference mode that chose it.
    """

    index: int
    chain: tuple[int, ...]
    payload: bytes
    picture_type: str
    reference: int | None
    mode: str

    @property
    def is_keyframe(self) -> bool:
        return len(self.chain) == 1

    @property
    def byte_count(self) -> int:
        return len(self.payload)


def _encode_chain(
    frames: Sequence[np.ndarray],
    *,
    codec: StreamCodec,
    crf: int,
) -> tuple[list[bytes], list[str], np.ndarray]:
    """Encode ``frames`` as one low-delay sequence; return packets and decode.

    Returns ``(packet_bytes, picture_types, decoded_frames)``. The decode is of
    the encoder's *own output*, which is the closed loop every video encoder
    already runs and the only reconstruction either side is allowed to hold.
    """
    ffmpeg_path, ffprobe = _ffmpeg()
    clip = np.ascontiguousarray(np.stack([np.asarray(f, dtype=np.uint8) for f in frames]))
    count, height, width, _ = clip.shape

    with tempfile.TemporaryDirectory(prefix="ps_bgstream_") as tmp_dir:
        tmp = Path(tmp_dir)
        source = tmp / "source.mkv"
        _run(
            [
                ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                "-framerate", "1", "-i", "-", "-c:v", "ffv1", str(source),
            ],
            clip.tobytes(),
        )
        stream = tmp / f"stream.{codec.container}"
        _run(
            [
                ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                "-c:v", codec.encoder, "-crf", str(crf), *codec.low_delay,
                # One keyframe, at the head. The chain is the GOP; a keyframe
                # inside it would make the payloads before it dead weight.
                "-g", "1000000", "-f", codec.container, str(stream),
            ],
            None,
        )
        blob = stream.read_bytes()
        packets = _probe_packets(ffprobe, stream, codec.container)
        decoded = _decode(ffmpeg_path, stream, codec.container, count, height, width)

    payloads = [blob[p.pos : p.pos + p.size] for p in packets]
    types = [p.picture_type for p in packets]
    if len(payloads) != count:
        raise StreamDrift(
            f"{codec.encoder} returned {len(payloads)} packets for {count} frames; "
            "a payload per scene is the whole accounting model"
        )
    if b"".join(payloads) != blob:
        raise StreamDrift(
            "packet slices do not reassemble the elementary stream, so a client "
            "concatenating payloads would not get back what the encoder emitted"
        )
    return payloads, types, decoded


@dataclass(frozen=True)
class _Packet:
    """One frame's byte range in the elementary stream, and its picture type."""

    pos: int
    size: int
    picture_type: str


def _probe_packets(ffprobe: str, path: Path, container: str) -> list[_Packet]:
    """Per-frame byte ranges and picture types, read off the elementary stream."""
    result = subprocess.run(
        [
            ffprobe, "-hide_banner", "-loglevel", "error", "-f", container,
            "-select_streams", "v:0",
            "-show_entries", "packet=pos,size:frame=pict_type",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True,
    )
    parsed = json.loads(result.stdout.decode("utf-8", "replace"))
    # Asking for packets *and* frames makes ffprobe emit one interleaved
    # `packets_and_frames` array rather than two lists. Reading it in order is
    # what pairs a picture type with the byte range it came from; zipping two
    # separate lists by position would silently mispair if either is short.
    entries = parsed.get("packets_and_frames")
    if entries is None:
        entries = parsed.get("packets", [])
    rows: list[_Packet] = []
    for entry in entries:
        if entry.get("type") == "frame":
            if rows:
                rows[-1] = replace(rows[-1], picture_type=str(entry.get("pict_type", "?")))
            continue
        rows.append(
            _Packet(pos=int(entry.get("pos", 0)), size=int(entry.get("size", 0)), picture_type="?")
        )
    return rows


def _decode(
    ffmpeg: str, path: Path, container: str, count: int, height: int, width: int
) -> np.ndarray:
    """Decode an elementary stream back to RGB.

    Names `-f <container>` on input. A decode that names no input format can
    fall back to probing, and findings §14 is what a second, unintended codec
    in the path costs: it capped every quality it touched.
    """
    raw = _run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-f", container, "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        None,
    )
    frame_bytes = height * width * 3
    usable = (len(raw) // frame_bytes) * frame_bytes
    decoded = np.frombuffer(raw[:usable], dtype=np.uint8).reshape(-1, height, width, 3)
    if decoded.shape[0] != count:
        raise StreamDrift(
            f"decoded {decoded.shape[0]} frames from a {count}-frame chain; "
            "a missing reconstruction is drift waiting to happen"
        )
    return decoded


def _run(argv: list[str], stdin_bytes: bytes | None) -> bytes:
    result = subprocess.run(argv, input=stdin_bytes, capture_output=True)
    if result.returncode != 0:
        detail = (result.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {detail[:400]}")
    return result.stdout


@dataclass(frozen=True)
class ChainEncode:
    """One chain's encode: the bytes per scene, the picture types, the decode.

    Args:
        payloads: One packet per frame, in order. Concatenated they are exactly
            the elementary stream, so a client can reassemble a prefix.
        picture_types: `I`/`P` per frame. A `B` means the encode was not causal.
        reconstructions: The decode of the encoder's own output -- the only
            pixels either side is allowed to predict from.
    """

    payloads: tuple[bytes, ...]
    picture_types: tuple[str, ...]
    reconstructions: np.ndarray

    @property
    def marginal_bytes(self) -> int:
        """Cost of the last scene given every scene before it in the chain."""
        return len(self.payloads[-1])


def encode_chain(
    frames: Sequence[np.ndarray], *, codec: str = "av1", crf: int = 38
) -> ChainEncode:
    """Encode a chain of scenes as one low-delay sequence.

    The public entry point for measurement code. A one-frame chain is a fresh
    intra -- the baseline amortisation has to beat -- and a two-frame chain is
    the trial encode findings §18 and §19 used, so numbers built on this sit on
    the same axis as the 31-53% already recorded.
    """
    if codec not in CODECS:
        raise ValueError(f"unknown stream codec {codec!r}; known: {sorted(CODECS)}")
    payloads, types, decoded = _encode_chain(frames, codec=CODECS[codec], crf=crf)
    return ChainEncode(
        payloads=tuple(payloads),
        picture_types=tuple(types),
        reconstructions=decoded,
    )


@dataclass
class BackgroundStreamTransmitter:
    """Encoder side: carries reconstructions across scenes.

    Args:
        mode: One of `REFERENCE_MODES`.
        codec: Key into `CODECS`.
        crf: Constant-quality rung. The same rung must be used on every arm
            compared, or the comparison is between rungs, not modes.
        keyframe_interval: Force a keyframe every *k* scenes under
            `periodic-i`. `KEYFRAME_NEVER` means a pure P-chain, which is the
            `never` column of the sweep -- no random access and no loss
            resilience, which brief §3 says is acceptable for a paper *as long
            as the paper says so*.
    """

    mode: str = REFERENCE_LAST
    codec: str = "av1"
    crf: int = 38
    keyframe_interval: int = KEYFRAME_NEVER

    _originals: list[np.ndarray] = field(default_factory=list, init=False)
    _chains: list[tuple[int, ...]] = field(default_factory=list, init=False)
    _payloads: list[bytes] = field(default_factory=list, init=False)
    _reconstructions: list[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.mode not in REFERENCE_MODES:
            raise ValueError(f"unknown reference mode {self.mode!r}; known: {list(REFERENCE_MODES)}")
        if self.codec not in CODECS:
            raise ValueError(f"unknown stream codec {self.codec!r}; known: {sorted(CODECS)}")
        if self.keyframe_interval < 0:
            raise ValueError("keyframe_interval must be >= 0; 0 means never")

    @property
    def spec(self) -> StreamCodec:
        return CODECS[self.codec]

    @property
    def reconstructions(self) -> tuple[np.ndarray, ...]:
        """What the encoder holds. Equal, frame for frame, to the client's."""
        return tuple(self._reconstructions)

    def reset(self) -> None:
        """Start a new independently coded background (a context boundary).

        The next ``push`` is a keyframe. A later context may use a different
        canvas size; that is not a shape error, because it is not the same
        stream.
        """
        self._originals.clear()
        self._chains.clear()
        self._payloads.clear()
        self._reconstructions.clear()

    def _forces_keyframe(self, index: int) -> bool:
        if index == 0:
            return True
        if self.mode != REFERENCE_PERIODIC_I or self.keyframe_interval == KEYFRAME_NEVER:
            return False
        return index % self.keyframe_interval == 0

    def _select_reference(self, plate: np.ndarray) -> int:
        """Which already-transmitted scene this one predicts from.

        Scored against the *reconstructions*, not the originals: the
        reconstruction is what the prediction actually starts from, and it is
        also the only thing the client has.
        """
        if self.mode == REFERENCE_FIRST:
            return 0
        if self.mode in (REFERENCE_LAST, REFERENCE_PERIODIC_I):
            return len(self._reconstructions) - 1
        scores = [canny_iou(plate, recon) for recon in self._reconstructions]
        return int(np.argmax(scores))

    def push(self, plate: np.ndarray) -> ScenePayload:
        """Transmit one scene's background and return its payload.

        The payload is final: prefix stability means no later scene can revise
        it, which is what makes each scene's cost independent of every future
        scene.
        """
        index = len(self._originals)
        array = np.asarray(plate, dtype=np.uint8)
        if self._originals and array.shape != self._originals[0].shape:
            raise ValueError(
                f"scene {index} is {array.shape}, the stream is {self._originals[0].shape}; "
                "inter prediction needs a fixed frame size (brief §2)"
            )
        self._originals.append(array)

        if self._forces_keyframe(index):
            reference: int | None = None
            chain: tuple[int, ...] = (index,)
        else:
            reference = self._select_reference(array)
            chain = self._chains[reference] + (index,)
        self._chains.append(chain)

        payloads, types, decoded = _encode_chain(
            [self._originals[i] for i in chain], codec=self.spec, crf=self.crf
        )
        self._assert_prefix_stable(chain, payloads)
        if "B" in types:
            raise StreamDrift(
                f"scene {index} encoded with a B-frame ({''.join(types)}); a B-frame "
                "references a future picture, so the payload is not causal"
            )

        self._payloads.append(payloads[-1])
        self._reconstructions.append(decoded[-1])
        return ScenePayload(
            index=index,
            chain=chain,
            payload=payloads[-1],
            picture_type=types[-1],
            reference=reference,
            mode=self.mode,
        )

    def export_state(self) -> dict[str, Any]:
        """Enough to continue the stream after a crash without re-encoding."""
        return {
            "mode": self.mode,
            "codec": self.codec,
            "crf": int(self.crf),
            "keyframe_interval": int(self.keyframe_interval),
            "chains": [list(chain) for chain in self._chains],
            "payloads": list(self._payloads),
            "originals": [np.asarray(item) for item in self._originals],
            "reconstructions": [np.asarray(item) for item in self._reconstructions],
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Restore ``export_state``. Refuses a codec/mode mismatch."""
        if str(state["mode"]) != self.mode or str(state["codec"]) != self.codec:
            raise ValueError(
                f"stream state is mode={state.get('mode')!r} codec={state.get('codec')!r}, "
                f"transmitter is {self.mode!r} {self.codec!r}"
            )
        if int(state["crf"]) != int(self.crf):
            raise ValueError("stream state crf does not match this transmitter")
        if int(state["keyframe_interval"]) != int(self.keyframe_interval):
            raise ValueError("stream state keyframe interval does not match this transmitter")
        self.reset()
        self._chains = [tuple(int(i) for i in chain) for chain in state["chains"]]
        self._payloads = [bytes(item) for item in state["payloads"]]
        self._originals = [np.asarray(item, dtype=np.uint8) for item in state["originals"]]
        self._reconstructions = [
            np.asarray(item, dtype=np.uint8) for item in state["reconstructions"]
        ]
        n = len(self._payloads)
        if not (len(self._chains) == len(self._originals) == len(self._reconstructions) == n):
            raise ValueError("stream state lists have unequal length")

    def _assert_prefix_stable(self, chain: tuple[int, ...], payloads: list[bytes]) -> None:
        """Re-derived payloads for earlier scenes must match what was sent.

        This is the check that would fire if lookahead or B-frames crept back
        in, and it is deliberately not a comment claiming the flags work.
        """
        for offset, scene in enumerate(chain[:-1]):
            if payloads[offset] != self._payloads[scene]:
                raise StreamDrift(
                    f"re-encoding the chain changed scene {scene}'s payload "
                    f"({len(self._payloads[scene])} -> {len(payloads[offset])} bytes). "
                    "The encode is not prefix-stable, so payloads are not causal."
                )


@dataclass
class BackgroundStreamReceiver:
    """Client side: holds the payloads it was sent and nothing else.

    It never sees an original plate, and it re-encodes nothing. Reconstructing
    scene *n* is decoding the payloads along scene *n*'s chain, which is the
    same bytes the encoder decoded -- so the two reconstructions are equal by
    construction.
    """

    codec: str = "av1"
    _payloads: dict[int, bytes] = field(default_factory=dict, init=False)
    _shape: tuple[int, int] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.codec not in CODECS:
            raise ValueError(f"unknown stream codec {self.codec!r}; known: {sorted(CODECS)}")

    def reset(self) -> None:
        """Drop received payloads. A context boundary starts a new decode chain."""
        self._payloads.clear()
        self._shape = None

    def import_payloads(self, payloads: Mapping[int, bytes]) -> None:
        """Restore previously received scene bytes without re-decoding them."""
        self.reset()
        self._payloads = {int(index): bytes(blob) for index, blob in payloads.items()}

    def receive(self, payload: ScenePayload, *, height: int, width: int) -> np.ndarray:
        """Accept one payload and return the plate the client now holds."""
        missing = [i for i in payload.chain[:-1] if i not in self._payloads]
        if missing:
            raise StreamDrift(
                f"scene {payload.index} predicts from {missing}, which the client "
                "never received; a P-chain has no random access (brief §3)"
            )
        self._payloads[payload.index] = payload.payload

        spec = CODECS[self.codec]
        blob = b"".join(self._payloads[i] for i in payload.chain)
        ffmpeg_path, _ = _ffmpeg()
        with tempfile.TemporaryDirectory(prefix="ps_bgrecv_") as tmp_dir:
            path = Path(tmp_dir) / f"chain.{spec.container}"
            path.write_bytes(blob)
            decoded = _decode(ffmpeg_path, path, spec.container, len(payload.chain), height, width)
        return decoded[-1]


def stream_linear(
    plates: Sequence[np.ndarray],
    *,
    codec: str = "av1",
    crf: int = 38,
    keyframe_interval: int = KEYFRAME_NEVER,
    mode: str = REFERENCE_LAST,
) -> list[ScenePayload]:
    """Payloads for a whole sequence whose chains are linear, in one pass.

    `last` and `periodic-i` both predict from the immediately preceding scene,
    so a scene's chain is the run of scenes back to its keyframe. Encoding that
    run once yields every payload in it -- **because the encode is
    prefix-stable**, the bytes read off for scene *i* here are the same bytes
    `BackgroundStreamTransmitter.push` would have emitted one scene at a time.
    That is the brief's "growing re-encode is equivalent" made exact rather than
    assumed, and `test_the_batch_path_agrees_with_pushing_scene_by_scene` is
    what keeps it honest.

    This exists for cost, not convenience: pushing scene by scene re-encodes the
    whole chain each time, which is O(N^2) frames over a sequence. At 4K that is
    the difference between a sweep that runs and one that does not.
    """
    if mode not in (REFERENCE_LAST, REFERENCE_PERIODIC_I):
        raise ValueError(
            f"{mode!r} does not have linear chains; use BackgroundStreamTransmitter.push"
        )
    spec = CODECS[codec]
    arrays = [np.asarray(p, dtype=np.uint8) for p in plates]
    if len({a.shape for a in arrays}) > 1:
        raise ValueError("the stream needs one frame size (brief §2)")

    forced = keyframe_interval if mode == REFERENCE_PERIODIC_I else KEYFRAME_NEVER
    groups: list[list[int]] = []
    for index in range(len(arrays)):
        if index == 0 or (forced != KEYFRAME_NEVER and index % forced == 0):
            groups.append([index])
        else:
            groups[-1].append(index)

    out: list[ScenePayload] = []
    for group in groups:
        payloads, types, _ = _encode_chain([arrays[i] for i in group], codec=spec, crf=crf)
        if "B" in types:
            raise StreamDrift(f"group {group} encoded with a B-frame ({''.join(types)}); not causal")
        for offset, scene in enumerate(group):
            out.append(
                ScenePayload(
                    index=scene,
                    chain=tuple(group[: offset + 1]),
                    payload=payloads[offset],
                    picture_type=types[offset],
                    reference=None if offset == 0 else group[offset - 1],
                    mode=mode,
                )
            )
    return out
