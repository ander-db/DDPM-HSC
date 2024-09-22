import pytest
import torch

from src.attention.spatial_attention import SpatialAttentionUNet


@pytest.fixture(params=[16, 32, 64, 128, 256, 512, 1024, 2048])
def g_channels(request):
    return request.param


@pytest.fixture
def attention_block(g_channels):
    return SpatialAttentionUNet(g_channels=g_channels)


@pytest.fixture
def input_tensors(g_channels):
    batch_size = 2
    height, width = 32, 32
    x = torch.randn(batch_size, g_channels // 2, height, width)
    g = torch.randn(batch_size, g_channels, height // 2, width // 2)
    return x, g


def test_spatial_attention_unet_output_shape(attention_block, input_tensors):
    x, g = input_tensors
    output, attention_map = attention_block(g, x)

    expected_output_shape = x.shape
    expected_attention_map_shape = (x.shape[0], 1, x.shape[2], x.shape[3])

    assert (
        output.shape == expected_output_shape
    ), f"Output shape {output.shape} does not match expected {expected_output_shape}"
    assert (
        attention_map.shape == expected_attention_map_shape
    ), f"Attention map shape {attention_map.shape} does not match expected {expected_attention_map_shape}"


def test_spatial_attention_unet_output_type(attention_block, input_tensors):
    x, g = input_tensors
    output, attention_map = attention_block(g, x)

    assert isinstance(output, torch.Tensor), "Output should be a PyTorch tensor"
    assert isinstance(
        attention_map, torch.Tensor
    ), "Attention map should be a PyTorch tensor"
    assert (
        output.dtype == x.dtype
    ), f"Output dtype {output.dtype} does not match input dtype {x.dtype}"
    assert (
        attention_map.dtype == x.dtype
    ), f"Attention map dtype {attention_map.dtype} does not match input dtype {x.dtype}"


def test_spatial_attention_unet_gradient_flow(attention_block, input_tensors):
    x, g = input_tensors
    x.requires_grad = True
    g.requires_grad = True

    output, attention_map = attention_block(g, x)
    loss = output.sum() + attention_map.sum()
    loss.backward()

    assert x.grad is not None, "No gradient flowing to x"
    assert g.grad is not None, "No gradient flowing to g"


def test_attention_map_range(attention_block, input_tensors):
    x, g = input_tensors
    _, attention_map = attention_block(g, x)

    assert torch.all(attention_map >= 0) and torch.all(
        attention_map <= 1
    ), "All attention map values must be between 0 and 1"
