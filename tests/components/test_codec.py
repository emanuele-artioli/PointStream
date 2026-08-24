"""Codec registry, command builder, capability rejection, tool recording.

Behaviour the caller relies on, and the misuse that would otherwise produce a
plausible-looking bitstream of the wrong format or the wrong encoder.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.components.codec import REGISTRY
from src.components.codec.command import build_command
from src.components.codec.tools import ResolvedTool, resolve_tool
from src.contracts import config
from src.contracts.codecs import EncodeRequest, RateControl
from src.contracts.errors import CodecConstraintError, ConfigError


def _tool(name: str = "ffmpeg", path: str = "/opt/local/bin/ffmpeg", *, features: tuple[str, ...] = ()) -> ResolvedTool:
    return ResolvedTool(name=name, path=path, version="test-version", features=frozenset(features))


def _request(codec_name: str, **kwargs: object) -> EncodeRequest:
    defaults: dict[str, object] = {
        "rate_control": RateControl.QP,
        "rate": 32,
        "preset": None,
        "pix_fmt": "yuv420p",
    }
    defaults.update(kwargs)
    return EncodeRequest(codec_name=codec_name, **defaults)  # type: ignore[arg-type]


class TestRegistryWiring:
    def test_the_four_rungs_are_registered_under_their_contract_names(self) -> None:
        assert REGISTRY.names() == ["av1", "avc", "hevc", "vvc"]
        assert REGISTRY.axis == "codec"

    def test_validate_backends_accepts_each_registered_rung(self) -> None:
        for name in ("avc", "hevc", "av1", "vvc"):
            loaded = config.load(
                {"fallback": {"codec": name, "rate_control": "qp", "rate": 32, "preset": None}}
            )
            config.validate_backends(loaded, registries={"codec": REGISTRY})

    def test_validate_backends_rejects_a_codec_missing_from_the_registry(self) -> None:
        """A config that names av1 is valid against the contract table; it is not
        valid against an empty component registry. That is the third pass."""
        from src.contracts.registry import Registry

        loaded = config.default()
        empty: Registry[object] = Registry("codec")
        with pytest.raises(ConfigError, match="av1"):
            config.validate_backends(loaded, registries={"codec": empty})


class TestCommandBuilder:
    def test_avc_qp_goes_through_ffmpeg_libx264(self) -> None:
        argv = build_command(
            "encode",
            _request("avc", preset="veryfast"),
            source="in.y4m",
            dest="out.mp4",
            encoder=_tool(),
        )
        assert argv[0] == "/opt/local/bin/ffmpeg"
        assert argv[argv.index("-c:v") + 1] == "libx264"
        assert argv[argv.index("-qp") + 1] == "32"
        assert argv[argv.index("-preset") + 1] == "veryfast"
        assert "libsvtav1" not in argv

    def test_hevc_is_kvazaar_not_ffmpeg(self) -> None:
        argv = build_command(
            "encode",
            _request("hevc", preset="ultrafast"),
            source="in.yuv",
            dest="out.hevc",
            encoder=_tool("kvazaar", "/opt/local/bin/kvazaar"),
            roi_file="roi.bin",
            width=128,
            height=64,
            fps=10,
        )
        assert argv[0] == "/opt/local/bin/kvazaar"
        assert argv[argv.index("--qp") + 1] == "32"
        assert argv[argv.index("--roi") + 1] == "roi.bin"
        assert argv[argv.index("--input-res") + 1] == "128x64"
        assert "-c:v" not in argv

    def test_av1_is_the_binary_with_roi_map_file(self) -> None:
        argv = build_command(
            "encode",
            _request("av1", preset="8"),
            source="in.y4m",
            dest="out.ivf",
            encoder=_tool("SvtAv1EncApp", "/opt/local/bin/SvtAv1EncApp", features=("roi-map-file",)),
            roi_file="roi.txt",
        )
        assert argv[0] == "/opt/local/bin/SvtAv1EncApp"
        assert argv[argv.index("--qp") + 1] == "32"
        assert argv[argv.index("--rc") + 1] == "0"
        assert "--roi-map-file" in argv
        assert argv[argv.index("--roi-map-file") + 1] == "roi.txt"
        assert "libsvtav1" not in argv
        assert "--aq-mode" not in argv

    def test_av1_qp_without_roi_forces_cqp(self) -> None:
        """--crf is --aq-mode 2. A QP request has to actually be CQP."""
        argv = build_command(
            "encode",
            _request("av1"),
            source="in.y4m",
            dest="out.ivf",
            encoder=_tool("SvtAv1EncApp", "/opt/local/bin/SvtAv1EncApp"),
        )
        assert argv[argv.index("--aq-mode") + 1] == "0"
        assert "--crf" not in argv

    def test_av1_bitrate_uses_bits_not_kilobits(self) -> None:
        argv = build_command(
            "encode",
            _request("av1", rate_control=RateControl.BITRATE, rate=400_000),
            source="in.y4m",
            dest="out.ivf",
            encoder=_tool("SvtAv1EncApp", "/opt/local/bin/SvtAv1EncApp"),
        )
        assert argv[argv.index("--tbr") + 1] == "400000b"

    def test_vvc_is_ffmpeg_libvvenc_with_qpa_off_under_qp(self) -> None:
        argv = build_command(
            "encode",
            _request("vvc", preset="faster"),
            source="in.y4m",
            dest="out.vvc",
            encoder=_tool(),
        )
        assert argv[argv.index("-c:v") + 1] == "libvvenc"
        assert argv[argv.index("-qp") + 1] == "32"
        assert argv[argv.index("-qpa") + 1] == "0"
        # libvvenc cannot emit yuv420p; lying about it would be the svt bug again.
        pix_idx = [i for i, token in enumerate(argv) if token == "-pix_fmt"]
        assert pix_idx == []

    def test_decode_is_ffmpeg_for_every_rung(self) -> None:
        argv = build_command(
            "decode",
            _request("hevc"),
            source="in.hevc",
            dest="out.y4m",
            encoder=_tool("kvazaar", "/opt/local/bin/kvazaar"),
            ffmpeg=_tool(),
        )
        assert argv[0] == "/opt/local/bin/ffmpeg"
        assert argv[argv.index("-pix_fmt") + 1] == "yuv420p"

    def test_avc_addroi_lands_in_the_filter_graph(self) -> None:
        argv = build_command(
            "encode",
            _request("avc"),
            source="in.y4m",
            dest="out.mp4",
            encoder=_tool(),
            addroi="addroi=16:16:32:32:-0.4",
        )
        assert "addroi=16:16:32:32:-0.4,format=yuv420p" in argv


class TestCapabilityRejection:
    def test_av1_yuv444p_raises_before_any_command_is_built(self) -> None:
        """libsvtav1 accepted this flag and emitted yuv420p. The binary path
        has to refuse it too, or the same ablation measures a dead knob."""
        with pytest.raises(CodecConstraintError, match="yuv444p"):
            build_command(
                "encode",
                EncodeRequest(codec_name="av1", rate_control=RateControl.QP, rate=32, pix_fmt="yuv444p"),
                source="in.y4m",
                dest="out.ivf",
                encoder=_tool("SvtAv1EncApp", "/opt/local/bin/SvtAv1EncApp"),
            )

    def test_libsvtav1_stuffed_into_extra_args_with_yuv444p_raises(self) -> None:
        with pytest.raises(CodecConstraintError, match="libsvtav1"):
            build_command(
                "encode",
                _request("avc", pix_fmt="yuv444p", extra_args=("-c:v", "libsvtav1")),
                source="in.y4m",
                dest="out.mp4",
                encoder=_tool(),
            )

    def test_vvc_native_roi_on_the_request_is_rejected_by_the_contract(self) -> None:
        with pytest.raises(CodecConstraintError, match="roi_map"):
            EncodeRequest(
                codec_name="vvc",
                rate_control=RateControl.QP,
                rate=32,
                roi_map="map.txt",
            ).validate()

    def test_av1_roi_without_the_flag_on_this_binary_names_path_and_version(self) -> None:
        encoder = _tool("SvtAv1EncApp", "/old/SvtAv1EncApp")  # no roi-map-file feature
        with pytest.raises(CodecConstraintError, match=r"/old/SvtAv1EncApp"):
            build_command(
                "encode",
                _request("av1"),
                source="in.y4m",
                dest="out.ivf",
                encoder=encoder,
                roi_file="roi.txt",
            )


class TestToolResolution:
    def test_env_path_and_version_are_what_get_recorded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        binary = tmp_path / "SvtAv1EncApp"
        binary.write_text("#!/bin/sh\n", encoding="ascii")
        monkeypatch.setenv("SVTAV1_BIN", str(binary))

        def fake_run(argv: list[str], **kwargs: object) -> object:
            class Result:
                stdout = ""
                stderr = ""

            if argv[1] == "--version":
                Result.stdout = "SVT-AV1 v1.8.0 (release)\n"
            elif argv[1] == "--help":
                Result.stdout = "  --roi-map-file   Enable ROI\n"
            return Result()

        with patch("src.components.codec.tools.subprocess.run", side_effect=fake_run):
            tool = resolve_tool("SVTAV1_BIN", "SvtAv1EncApp")

        assert tool.path == str(binary)
        assert tool.version == "SVT-AV1 v1.8.0 (release)"
        assert tool.has("roi-map-file")

    def test_a_missing_explicit_path_is_an_error_not_a_silent_path_lookup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        missing = tmp_path / "not-here" / "SvtAv1EncApp"
        monkeypatch.setenv("SVTAV1_BIN", str(missing))
        with pytest.raises(FileNotFoundError, match="SVTAV1_BIN"):
            resolve_tool("SVTAV1_BIN", "SvtAv1EncApp")


def test_run_retries_when_the_encoder_writes_an_empty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """libvvenc has exited 0 after a 0-byte 4K QP-48 bitstream more than once."""
    from src.components.codec import encode as encode_mod

    dest = tmp_path / "out.vvc"
    calls = {"n": 0}

    def fake_run(argv: list[str], **kwargs: object) -> object:
        del argv, kwargs
        calls["n"] += 1
        dest.write_bytes(b"" if calls["n"] == 1 else b"vvc-bytes")

        class Result:
            returncode = 0
            stderr = ""
            stdout = ""

        return Result()

    monkeypatch.setattr(encode_mod.subprocess, "run", fake_run)
    encode_mod._run(["ffmpeg", "-y", str(dest)], dest)
    assert calls["n"] == 2
    assert dest.read_bytes() == b"vvc-bytes"
