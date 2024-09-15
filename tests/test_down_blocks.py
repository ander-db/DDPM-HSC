import pytest
import torch
import torch.nn as nn

from src.down_blocks import DownsampleDoubleResBlock
from src.res_blocks import ResBlockGroupNorm, ResBlockBatchNorm


@pytest.fixture(params=[DownsampleDoubleResBlock])
def down_block(request):
    return request.param(64, 128)


def test_initialization(down_block):
    assert down_block.in_channels == 64
    assert down_block.out_channels == 128


def test_structure(down_block):
    assert isinstance(down_block.down_block, nn.Sequential)
    assert isinstance(down_block.down_block[0], nn.MaxPool2d)


def test_output_shape(down_block):
    x = torch.randn(1, 64, 32, 32)
    output = down_block(x)
    assert output.shape == (1, 128, 16, 16)


@pytest.mark.parametrize(
    "input_shape", [(1, 64, 32, 32), (2, 64, 64, 64), (4, 64, 128, 128)]
)
def test_different_input_sizes(down_block, input_shape):
    x = torch.randn(*input_shape)
    output = down_block(x)
    expected_shape = (input_shape[0], 128, input_shape[2] // 2, input_shape[3] // 2)
    assert output.shape == expected_shape


def test_grad_flow(down_block):
    x = torch.randn(1, 64, 32, 32, requires_grad=True)
    output = down_block(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    for param in down_block.parameters():
        assert param.grad is not None


def test_resblock_types():
    block_bn = DownsampleDoubleResBlock(64, 128, res_block_type=ResBlockBatchNorm)
    block_gn = DownsampleDoubleResBlock(64, 128, res_block_type=ResBlockGroupNorm)
    assert isinstance(block_bn.down_block[1], ResBlockBatchNorm)
    assert isinstance(block_gn.down_block[1], ResBlockGroupNorm)


def test_custom_params():
    block = DownsampleDoubleResBlock(
        64, 128, res_block_type=ResBlockGroupNorm, res_block_params={"n_groups": 16}
    )
    assert block.down_block[1].n_groups == 16
    assert block.down_block[2].n_groups == 16


def test_forward_pass():
    block = DownsampleDoubleResBlock(64, 128)
    x = torch.randn(1, 64, 32, 32)
    output = block(x)
    assert output.shape == (1, 128, 16, 16)
    assert torch.isfinite(output).all()
