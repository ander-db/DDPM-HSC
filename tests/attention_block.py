import pytest
import torch
import torch.nn as nn

from src.model import AttentionBlock


@pytest.fixture(params=[32, 64, 128])
def base_channels(request):
    return request.param


@pytest.fixture(params=[0.01, 0.1, 0.5])
def dropout_rate(request):
    return request.param


@pytest.fixture
def attention_block(base_channels, dropout_rate):
    return AttentionBlock(base_channels=base_channels, dropout_rate=dropout_rate)


@pytest.mark.parametrize(
    "batch_size,height,width",
    [
        (1, 32, 32),
        (4, 64, 64),
        (8, 16, 16),
    ],
)
def test_output_shape(attention_block, base_channels, batch_size, height, width):
    x = torch.randn(batch_size, base_channels * 2, height, width)
    g = torch.randn(batch_size, base_channels, height // 2, width // 2)

    output = attention_block(x, g)

    expected_shape = (batch_size, base_channels, height, width)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("height,width", [(16, 16), (32, 32), (64, 64)])
def test_different_input_sizes(
    attention_block, base_channels, batch_size, height, width
):
    x = torch.randn(batch_size, base_channels * 2, height, width)
    g = torch.randn(batch_size, base_channels, height // 2, width // 2)

    output = attention_block(x, g)

    expected_shape = (batch_size, base_channels, height, width)
    assert (
        output.shape == expected_shape
    ), f"Failed for input shapes: x={x.shape}, g={g.shape}"


def test_attention_effect(attention_block, base_channels):
    batch_size, height, width = 1, 32, 32
    x = torch.ones(batch_size, base_channels * 2, height, width)
    g = torch.ones(batch_size, base_channels, height // 2, width // 2)

    output = attention_block(x, g)

    assert not torch.allclose(
        output, torch.ones_like(output)
    ), "Attention should modify the input, output should not be all ones"


def test_dropout_effect(attention_block, base_channels, dropout_rate):
    if dropout_rate == 0.0:
        pytest.skip("Skipping dropout test for dropout_rate=0.0")

    batch_size, height, width = 1, 32, 32
    x = torch.ones(batch_size, base_channels * 2, height, width)
    g = torch.ones(batch_size, base_channels, height // 2, width // 2)

    attention_block.train()
    output1 = attention_block(x, g)
    output2 = attention_block(x, g)

    assert not torch.allclose(
        output1, output2
    ), "Dropout should cause different outputs in training mode"


def test_no_dropout_in_eval_mode(attention_block, base_channels):
    batch_size, height, width = 1, 32, 32
    x = torch.ones(batch_size, base_channels * 2, height, width)
    g = torch.ones(batch_size, base_channels, height // 2, width // 2)

    attention_block.eval()
    output1 = attention_block(x, g)
    output2 = attention_block(x, g)

    assert torch.allclose(
        output1, output2
    ), "Outputs should be the same in eval mode (no dropout)"


def test_grad_flow(attention_block, base_channels):
    batch_size, height, width = 1, 32, 32
    x = torch.randn(batch_size, base_channels * 2, height, width, requires_grad=True)
    g = torch.randn(
        batch_size, base_channels, height // 2, width // 2, requires_grad=True
    )

    output = attention_block(x, g)
    loss = output.sum()
    loss.backward()

    assert (
        x.grad is not None and g.grad is not None
    ), "Gradients should flow back to both inputs"

    for name, param in attention_block.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
