import pytest
import torch

from ltx_video.models.autoencoders.dual_conv3d import DualConv3d


@pytest.mark.parametrize("padding_mode", ["zeros", "reflect", "replicate", "circular"])
def test_dual_conv3d_paths_preserve_padding_mode(padding_mode):
    """The optimized 2-D/1-D path must match the native 3-D path."""
    torch.manual_seed(0)
    layer = DualConv3d(
        in_channels=3,
        out_channels=5,
        kernel_size=(3, 3, 3),
        stride=(2, 2, 2),
        padding=(1, 1, 1),
        padding_mode=padding_mode,
    )
    sample = torch.randn(2, 3, 5, 8, 8, requires_grad=True)

    native = layer(sample, use_conv3d=True)
    factorized = layer(sample, use_conv3d=False)

    assert native.shape == factorized.shape
    torch.testing.assert_close(native, factorized, atol=1e-6, rtol=1e-5)


def test_dual_conv3d_factorized_path_backpropagates_with_replicate_padding():
    """Non-zero padding must remain differentiable in the VAE fast path."""
    torch.manual_seed(1)
    layer = DualConv3d(
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        stride=(2, 1, 1),
        padding=(1, 1, 1),
        padding_mode="replicate",
    )
    sample = torch.randn(2, 4, 7, 6, 6, requires_grad=True)

    loss = layer(sample).square().mean()
    loss.backward()

    assert sample.grad is not None
    assert torch.isfinite(sample.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_dual_conv3d_rejects_unknown_padding_mode():
    with pytest.raises(ValueError, match="padding_mode"):
        DualConv3d(1, 1, kernel_size=3, padding_mode="invalid")
