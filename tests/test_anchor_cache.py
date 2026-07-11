from __future__ import annotations

from pathlib import Path

from src.encoder import anchor_cache


def _make_video(tmp_path: Path, name: str = "video.mp4", content: bytes = b"v" * 100) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


class TestAnchorCacheKey:
    def test_same_inputs_produce_same_key(self, tmp_path: Path) -> None:
        video = _make_video(tmp_path)
        key1 = anchor_cache.anchor_cache_key(video, 0.0, 2.0, "libsvtav1", 35, "fast")
        key2 = anchor_cache.anchor_cache_key(video, 0.0, 2.0, "libsvtav1", 35, "fast")
        assert key1 == key2

    def test_different_span_produces_different_key(self, tmp_path: Path) -> None:
        video = _make_video(tmp_path)
        key1 = anchor_cache.anchor_cache_key(video, 0.0, 2.0, "libsvtav1", 35, "fast")
        key2 = anchor_cache.anchor_cache_key(video, 2.0, 4.0, "libsvtav1", 35, "fast")
        assert key1 != key2

    def test_different_codec_settings_produce_different_key(self, tmp_path: Path) -> None:
        video = _make_video(tmp_path)
        key1 = anchor_cache.anchor_cache_key(video, 0.0, 2.0, "libsvtav1", 35, "fast")
        key2 = anchor_cache.anchor_cache_key(video, 0.0, 2.0, "libx265", 35, "fast")
        key3 = anchor_cache.anchor_cache_key(video, 0.0, 2.0, "libsvtav1", 20, "fast")
        assert len({key1, key2, key3}) == 3

    def test_different_video_content_produces_different_key(self, tmp_path: Path) -> None:
        video_a = _make_video(tmp_path, "a.mp4", content=b"a" * 100)
        video_b = _make_video(tmp_path, "b.mp4", content=b"b" * 200)
        key_a = anchor_cache.anchor_cache_key(video_a, 0.0, 2.0, "libsvtav1", 35, "fast")
        key_b = anchor_cache.anchor_cache_key(video_b, 0.0, 2.0, "libsvtav1", 35, "fast")
        assert key_a != key_b


class TestGetOrEncode:
    def test_cache_miss_calls_encode_fn_and_populates_cache(self, tmp_path: Path) -> None:
        video = _make_video(tmp_path)
        cache_root = tmp_path / "cache"
        output_path = tmp_path / "out.mp4"
        calls: list[Path] = []

        def _encode(dest: Path) -> None:
            calls.append(dest)
            dest.write_bytes(b"x" * 42)

        size, cache_hit = anchor_cache.get_or_encode(
            cache_root=cache_root,
            video_path=video,
            t_start=0.0,
            t_end=2.0,
            codec="libsvtav1",
            crf=35,
            preset="fast",
            output_path=output_path,
            encode_fn=_encode,
        )

        assert size == 42
        assert cache_hit is False
        assert len(calls) == 1
        key = anchor_cache.anchor_cache_key(video, 0.0, 2.0, "libsvtav1", 35, "fast")
        assert anchor_cache.cached_anchor_path(cache_root, key).exists()

    def test_cache_hit_skips_encode_fn(self, tmp_path: Path) -> None:
        video = _make_video(tmp_path)
        cache_root = tmp_path / "cache"
        first_output = tmp_path / "first.mp4"
        second_output = tmp_path / "second.mp4"
        call_count = {"n": 0}

        def _encode(dest: Path) -> None:
            call_count["n"] += 1
            dest.write_bytes(b"y" * 99)

        anchor_cache.get_or_encode(
            cache_root=cache_root, video_path=video, t_start=0.0, t_end=2.0,
            codec="libsvtav1", crf=35, preset="fast", output_path=first_output, encode_fn=_encode,
        )
        size, cache_hit = anchor_cache.get_or_encode(
            cache_root=cache_root, video_path=video, t_start=0.0, t_end=2.0,
            codec="libsvtav1", crf=35, preset="fast", output_path=second_output, encode_fn=_encode,
        )

        assert call_count["n"] == 1  # not called again on the hit
        assert cache_hit is True
        assert size == 99
        assert second_output.read_bytes() == b"y" * 99

    def test_different_span_is_a_separate_cache_miss(self, tmp_path: Path) -> None:
        video = _make_video(tmp_path)
        cache_root = tmp_path / "cache"
        call_count = {"n": 0}

        def _encode(dest: Path) -> None:
            call_count["n"] += 1
            dest.write_bytes(b"z" * 10)

        anchor_cache.get_or_encode(
            cache_root=cache_root, video_path=video, t_start=0.0, t_end=2.0,
            codec="libsvtav1", crf=35, preset="fast", output_path=tmp_path / "o1.mp4", encode_fn=_encode,
        )
        anchor_cache.get_or_encode(
            cache_root=cache_root, video_path=video, t_start=2.0, t_end=4.0,
            codec="libsvtav1", crf=35, preset="fast", output_path=tmp_path / "o2.mp4", encode_fn=_encode,
        )

        assert call_count["n"] == 2
