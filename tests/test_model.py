"""Model architecture: forward pass, shapes, configurability, gradient flow."""
import pytest
import torch

from src.models import (UNet, ResidualUNet, AttentionUNet, ConfigurableUNet,
                        MODEL_REGISTRY, build_model)
from src.config import load_config


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_forward_pass_preserves_spatial_dimensions(name):
    model = MODEL_REGISTRY[name](in_channels=1, out_channels=1, base_channels=8)
    out = model(torch.randn(2, 1, 64, 64))
    assert out.shape == (2, 1, 64, 64)


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_forward_pass_handles_non_power_of_two_size(name):
    """The padding logic in the decoder must handle sizes that don't halve cleanly."""
    model = MODEL_REGISTRY[name](in_channels=1, out_channels=1, base_channels=8)
    out = model(torch.randn(1, 1, 130, 130))
    assert out.shape == (1, 1, 130, 130)


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_transposed_convolution_upsampling_works(name):
    model = MODEL_REGISTRY[name](base_channels=8, bilinear=False)
    assert model(torch.randn(1, 1, 64, 64)).shape == (1, 1, 64, 64)


def test_output_is_raw_logits_not_probabilities():
    """The model must emit logits; the sigmoid lives in the loss and in metrics."""
    torch.manual_seed(0)
    out = UNet(base_channels=8)(torch.randn(4, 1, 64, 64) * 5)
    assert out.min() < 0.0, "Output looks bounded below at 0 — is a sigmoid baked in?"


def test_configurable_channels_and_classes():
    model = UNet(in_channels=3, out_channels=4, base_channels=8)
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 4, 64, 64)


def test_base_channels_controls_capacity():
    small = sum(p.numel() for p in UNet(base_channels=8).parameters())
    large = sum(p.numel() for p in UNet(base_channels=16).parameters())
    assert large > small * 2


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_gradients_reach_the_first_layer(name):
    """Catches a broken skip connection or a detached branch."""
    model = MODEL_REGISTRY[name](base_channels=8)
    out = model(torch.randn(1, 1, 64, 64))
    out.mean().backward()
    first = next(model.in_conv.parameters())
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert first.grad.abs().sum() > 0


def test_configurable_unet_matches_plain_unet_parameter_count():
    """residual=False, attention=False must reproduce the baseline topology."""
    a = sum(p.numel() for p in UNet(base_channels=8).parameters())
    b = sum(p.numel() for p in ConfigurableUNet(
        base_channels=8, residual_blocks=False, attention_gates=False).parameters())
    assert a == b


def test_variants_add_parameters_over_the_baseline():
    base = sum(p.numel() for p in UNet(base_channels=8).parameters())
    assert sum(p.numel() for p in ResidualUNet(base_channels=8).parameters()) > base
    assert sum(p.numel() for p in AttentionUNet(base_channels=8).parameters()) > base


def test_build_model_uses_config_name():
    cfg = load_config()
    assert isinstance(build_model(cfg), UNet)


def test_build_model_honors_override():
    cfg = load_config()
    assert isinstance(build_model(cfg, name_override="attention_unet"), AttentionUNet)


def test_build_model_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown model name"):
        build_model(load_config(), name_override="transformer_supreme")


def test_eval_mode_is_deterministic():
    model = UNet(base_channels=8).eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))
