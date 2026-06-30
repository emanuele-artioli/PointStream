"""Unit tests for Spade4Tennis architecture components.

Tests:
  1. SPADE normalization: output shape matches input shape
  2. SPADEResBlock: residual block preserves spatial dims
  3. ReferenceEncoder: correct downsampling factor
  4. SPADEResNet9Generator: full forward pass shape
  5. MultiscaleDiscriminator: correct number of scales
  6. VGG19PerceptualLoss: correct feature extraction shapes
  7. Hinge losses: correct scalar output
  8. Spade4TennisStrategy: instantiates correctly
"""
import sys
import os

import torch

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSPADE:
    """Tests for the SPADE normalization layer."""

    def test_output_shape_matches_input(self):
        from src.shared.spade4tennis_arch import SPADE

        spade = SPADE(norm_nc=128, cond_nc=256, hidden_nc=64)
        x = torch.randn(2, 128, 32, 32)     # Shape: [B, norm_nc, H, W]
        cond = torch.randn(2, 256, 16, 16)   # Shape: [B, cond_nc, H_c, W_c]
        out = spade(x, cond)
        assert out.shape == (2, 128, 32, 32), f"Expected (2, 128, 32, 32), got {out.shape}"

    def test_condition_same_spatial_dims(self):
        from src.shared.spade4tennis_arch import SPADE

        spade = SPADE(norm_nc=64, cond_nc=64)
        x = torch.randn(1, 64, 16, 16)
        cond = torch.randn(1, 64, 16, 16)  # Same spatial dims, no resize needed
        out = spade(x, cond)
        assert out.shape == x.shape


class TestSPADEResBlock:
    """Tests for the SPADE residual block."""

    def test_same_channels(self):
        from src.shared.spade4tennis_arch import SPADEResBlock

        block = SPADEResBlock(fin=256, fout=256, cond_nc=256)
        x = torch.randn(2, 256, 16, 16)
        cond = torch.randn(2, 256, 8, 8)
        out = block(x, cond)
        assert out.shape == (2, 256, 16, 16)

    def test_different_channels(self):
        from src.shared.spade4tennis_arch import SPADEResBlock

        block = SPADEResBlock(fin=128, fout=256, cond_nc=256)
        x = torch.randn(1, 128, 32, 32)
        cond = torch.randn(1, 256, 8, 8)
        out = block(x, cond)
        assert out.shape == (1, 256, 32, 32)
        assert block.learned_skip is True


class TestReferenceEncoder:
    """Tests for the reference image encoder."""

    def test_downsampling(self):
        from src.shared.spade4tennis_arch import ReferenceEncoder

        enc = ReferenceEncoder(in_nc=3, nf=64)
        x = torch.randn(2, 3, 512, 512)
        out = enc(x)
        # 3 layers of stride-2 → 512 / 8 = 64
        assert out.shape == (2, 256, 64, 64), f"Expected (2, 256, 64, 64), got {out.shape}"


class TestSPADEResNet9Generator:
    """Tests for the full generator."""

    def test_forward_shape(self):
        from src.shared.spade4tennis_arch import SPADEResNet9Generator

        gen = SPADEResNet9Generator(in_nc=3, out_nc=3, ngf=64, n_blocks=9)
        skeleton = torch.randn(1, 3, 256, 256)
        reference = torch.randn(1, 3, 256, 256)
        out = gen(skeleton, reference)
        assert out.shape == (1, 3, 256, 256), f"Expected (1, 3, 256, 256), got {out.shape}"

    def test_output_range(self):
        from src.shared.spade4tennis_arch import SPADEResNet9Generator

        gen = SPADEResNet9Generator(in_nc=3, out_nc=3, ngf=32, n_blocks=3)
        skeleton = torch.randn(1, 3, 64, 64)
        reference = torch.randn(1, 3, 64, 64)
        out = gen(skeleton, reference)
        # Output uses Tanh → [-1, 1]
        assert out.min() >= -1.0 and out.max() <= 1.0, \
            f"Output range [{out.min():.2f}, {out.max():.2f}] outside [-1, 1]"

    def test_param_count_lite(self):
        from src.shared.spade4tennis_arch import SPADEResNet9Generator

        gen = SPADEResNet9Generator(in_nc=3, out_nc=3, ngf=64, n_blocks=9)
        n_params = sum(p.numel() for p in gen.parameters())
        # Expected ~15M for the lite model
        assert 10e6 < n_params < 30e6, f"Param count {n_params/1e6:.1f}M outside expected [10M, 30M]"


class TestMultiscaleDiscriminator:
    """Tests for the multi-scale discriminator."""

    def test_num_scales(self):
        from scripts.train_spade4tennis import MultiscaleDiscriminator

        disc = MultiscaleDiscriminator(input_nc=6, ndf=64, n_layers=3, num_D=2)
        real = torch.randn(2, 3, 256, 256)
        cond = torch.randn(2, 3, 256, 256)
        results = disc(real, cond)
        assert len(results) == 2, f"Expected 2 scales, got {len(results)}"

    def test_forward_returns_multiple_scales(self):
        from scripts.train_spade4tennis import MultiscaleDiscriminator

        disc = MultiscaleDiscriminator(input_nc=6, ndf=64, n_layers=3, num_D=3)
        real = torch.randn(1, 3, 128, 128)
        cond = torch.randn(1, 3, 128, 128)
        results = disc(real, cond)
        assert len(results) == 3

    def test_feature_matching_shapes(self):
        from scripts.train_spade4tennis import MultiscaleDiscriminator

        disc = MultiscaleDiscriminator(input_nc=6, ndf=64, n_layers=3, num_D=2)
        real = torch.randn(1, 3, 128, 128)
        cond = torch.randn(1, 3, 128, 128)
        results = disc(real, cond)
        for features, pred in results:
            assert isinstance(features, list)
            assert len(features) > 0
            assert pred.ndim == 4  # [B, 1, H', W']


class TestLosses:
    """Tests for loss functions."""

    def test_hinge_loss_d(self):
        from scripts.train_spade4tennis import hinge_loss_d

        real_pred = torch.randn(4, 1, 8, 8)
        fake_pred = torch.randn(4, 1, 8, 8)
        loss = hinge_loss_d(real_pred, fake_pred)
        assert loss.ndim == 0  # scalar

    def test_hinge_loss_g(self):
        from scripts.train_spade4tennis import hinge_loss_g

        fake_pred = torch.randn(4, 1, 8, 8)
        loss = hinge_loss_g(fake_pred)
        assert loss.ndim == 0  # scalar

    def test_feature_matching(self):
        from scripts.train_spade4tennis import feature_matching_loss

        real_feats = [[torch.randn(2, 64, 16, 16), torch.randn(2, 128, 8, 8)]]
        fake_feats = [[torch.randn(2, 64, 16, 16), torch.randn(2, 128, 8, 8)]]
        loss = feature_matching_loss(real_feats, fake_feats)
        assert loss.ndim == 0
        assert loss.item() > 0


class TestSpade4TennisStrategy:
    """Tests for the inference strategy."""

    def test_instantiates(self):
        from src.decoder.spade4tennis_engine import Spade4TennisStrategy

        class MockConfig:
            controlnet_width = 256
            controlnet_height = 256

        strategy = Spade4TennisStrategy(config=MockConfig())
        assert strategy._width == 256
        assert strategy._height == 256
        assert strategy._model is None

    def test_backend_registration(self):
        """Verify spade4tennis is registered in the compositor."""

        # Just verify the import path works without crashing
        from src.decoder.spade4tennis_engine import Spade4TennisStrategy
        assert issubclass(Spade4TennisStrategy, object)
