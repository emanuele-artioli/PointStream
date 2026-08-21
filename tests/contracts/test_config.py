"""The configuration document, and what it refuses.

Each test here names a misconfiguration that would otherwise run and produce a
number nobody could tell was wrong.
"""

from __future__ import annotations

import pytest

from src.contracts import codecs, config, domain, keypoints, lattice, parsing
from src.contracts.errors import ConfigError


class TestTheDefault:
    def test_the_shipped_default_validates(self) -> None:
        """A default that fails its own validation is worse than no default."""
        config.validate(config.default())

    def test_the_default_names_a_real_domain_and_schema(self) -> None:
        default = config.default()
        assert domain.profile(default.domain).name == "tennis"
        assert keypoints.schema(default.pose.schema) is keypoints.CANONICAL_HUMAN

    def test_the_default_round_trips_through_its_own_renderer(self) -> None:
        """`config/default.yaml` is generated from the schema, not maintained.

        That is what makes a documented-but-unreachable key impossible: a key can
        only appear in the file if a field exists to produce it.
        """
        rendered = config.render_default()
        assert config.parse(rendered) == config.default()

    def test_generation_defaults_off_because_no_generator_is_named(self) -> None:
        default = config.default()
        assert default.generator.backend == "none"
        assert lattice.STAGE_GENERATION not in default.stages.enabled


class TestCodecConstraints:
    def test_a_pixel_format_the_codec_ignores_is_rejected(self) -> None:
        """SVT-AV1 accepts the flag and emits yuv420p anyway.

        Every residual encode since the knob was added silently requested a
        format it never got, and the ablation built on it measured nothing.
        """
        with pytest.raises(ConfigError, match="yuv444p"):
            config.load({"residual": {"codec": "av1", "pix_fmt": "yuv444p"}})

    def test_region_of_interest_on_a_codec_without_one_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="region-of-interest"):
            config.load(
                {"fallback": {"codec": "vvc", "roi": True, "rate_control": "qp", "rate": 32}}
            )

    def test_a_rate_control_the_codec_lacks_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="rate_control"):
            config.load({"fallback": {"codec": "vvc", "rate_control": "crf", "rate": 35}})

    def test_a_valid_roi_arm_is_accepted(self) -> None:
        loaded = config.load(
            {"fallback": {"codec": "av1", "roi": True, "rate_control": "qp", "rate": 40}}
        )
        assert loaded.fallback.roi
        loaded.fallback.encode_request(roi_map="map.txt").validate()


class TestDomainConstraints:
    def test_a_panorama_under_a_free_moving_camera_is_rejected(self) -> None:
        """DAVIS clips are handheld. A panorama built there is quietly wrong.

        Not merely worse — parallax means no single homography relates the
        frames, so the plate cannot be internally consistent at all.
        """
        with pytest.raises(ConfigError, match="parallax"):
            config.load({"domain": "general"})

    def test_the_general_domain_works_without_a_panorama(self) -> None:
        loaded = config.load(
            {"domain": "general", "background": {"method": domain.BACKGROUND_NONE}}
        )
        assert loaded.profile.name == "general"

    def test_an_unknown_domain_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            config.load({"domain": "underwater-basket-weaving"})


class TestMotionIsPerClass:
    def test_a_class_without_a_skeleton_falls_back_and_says_so(self) -> None:
        """A racket has no joints. A global `keypoints` setting cannot be right
        for every class at once, and a silent substitution would read as a
        quality loss rather than a misconfiguration."""
        resolved = config.default().motion.resolve(domain.profile("tennis"))

        assert resolved.by_class["player"] == "keypoints"
        assert resolved.by_class["racket"] == "encoded-video"
        assert "racket" in resolved.fell_back
        assert "player" not in resolved.fell_back

    def test_an_explicit_override_onto_a_skeletonless_class_is_rejected(self) -> None:
        """Falling back is fine; being *told* to do the impossible is not."""
        with pytest.raises(ConfigError, match="no keypoint schema"):
            config.load({"motion": {"per_class": {"racket": "keypoints"}}})

    def test_an_explicit_override_the_class_supports_is_accepted(self) -> None:
        loaded = config.load({"motion": {"per_class": {"racket": "motion-vectors"}}})
        resolved = loaded.motion.resolve(loaded.profile)
        assert resolved.by_class["racket"] == "motion-vectors"
        assert "racket" not in resolved.fell_back


class TestLatticeCoherence:
    def test_a_named_generator_with_the_stage_off_is_rejected(self) -> None:
        """It would never run, and the run would look like it had one."""
        with pytest.raises(ConfigError, match="never run"):
            config.load(
                {
                    "generator": {"backend": "controlnet", "variant": "canny"},
                    "lattice": {"generation": False},
                }
            )

    def test_the_stage_on_with_no_generator_named_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="no generator is named"):
            config.load({"lattice": {"generation": True}})

    def test_a_coherent_generative_config_is_accepted(self) -> None:
        loaded = config.load(
            {
                "generator": {"backend": "controlnet", "variant": "canny"},
                "lattice": {"generation": True},
            }
        )
        assert loaded.generator.resolved_name == "canny-controlnet"
        assert lattice.STAGE_GENERATION in loaded.stages.enabled

    def test_metrics_cannot_all_be_switched_off(self) -> None:
        """Quality is measured in every configuration, without exception."""
        with pytest.raises(ConfigError):
            config.load({"evaluation": {"metrics": []}})

    def test_the_all_off_corner_is_expressible(self) -> None:
        """Turn everything off and what is left is the source video — the
        baseline every component is measured against."""
        off = dict.fromkeys(
            (name for name in parsing.flat_keys(config.LatticeConfig)), False
        )
        loaded = config.load({"lattice": off, "background": {"method": domain.BACKGROUND_NONE}})
        assert loaded.stages.enabled == lattice.REQUIRED_STAGES


class TestProblemsAreReportedTogether:
    def test_several_independent_problems_all_surface(self) -> None:
        with pytest.raises(ConfigError) as caught:
            config.load(
                {
                    "residual": {"codec": "av1", "pix_fmt": "yuv444p"},
                    "motion": {"per_class": {"ball": "keypoints"}},
                    "lattice": {"generation": True},
                }
            )
        assert len(caught.value.problems) == 3

    def test_a_typo_is_an_error_not_a_no_op(self) -> None:
        with pytest.raises(ConfigError, match="Did you mean"):
            config.load({"residual": {"pix-fmtt": "yuv420p"}})


class TestEncodeRequests:
    def test_the_residual_request_reflects_the_section(self) -> None:
        loaded = config.load({"residual": {"codec": "hevc", "rate_control": "qp", "rate": 30}})
        request = loaded.residual.encode_request()
        assert request.codec_name == "hevc"
        assert request.rate_control is codecs.RateControl.QP
        request.validate()
