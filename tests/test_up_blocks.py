import pytest
import torch
import torch.nn as nn

from src.up_blocks import UpSampleDoubleResBlock, SingleConvUpBlock
from src.res_blocks import ResBlockBatchNorm, ResBlockGroupNorm


@pytest.fixture(params=[UpSampleDoubleResBlock, SingleConvUpBlock])
def up_block(request):
    if request.param == UpSampleDoubleResBlock:
        return request.param(64, 32)
    else:
        return request.param(64, 32)


def test_initialization(up_block):
    assert up_block.in_channels == 64
    assert up_block.out_channels == 32


def test_structure(up_block):
    assert hasattr(up_block, "upsample")
    assert isinstance(up_block.upsample, nn.ConvTranspose2d)

    if isinstance(up_block, UpSampleDoubleResBlock):
        assert hasattr(up_block, "conv_blocks")
        assert isinstance(up_block.conv_blocks, nn.Sequential)
    elif isinstance(up_block, SingleConvUpBlock):
        assert hasattr(up_block, "conv")
        assert isinstance(up_block.conv, nn.Conv2d)


def test_output_shape(up_block):
    x = torch.randn(1, 64, 16, 16)
    skip = torch.randn(1, 32, 32, 32)
    output = up_block(x, skip)
    assert output.shape == (1, 32, 32, 32)


@pytest.mark.parametrize(
    "input_shape,skip_shape",
    [
        ((1, 64, 16, 16), (1, 32, 32, 32)),
        ((2, 64, 32, 32), (2, 32, 64, 64)),
        ((4, 64, 8, 8), (4, 32, 16, 16)),
    ],
)
def test_different_input_sizes(up_block, input_shape, skip_shape):
    print(f'[INFO] input_shape: {input_shape}, skip_shape: {skip_shape}')
    x = torch.randn(*input_shape)
    skip = torch.randn(*skip_shape)
    output = up_block(x, skip)
    assert output.shape == skip_shape


def test_grad_flow(up_block):
    x = torch.randn(1, 64, 16, 16, requires_grad=True)
    skip = torch.randn(1, 32, 32, 32, requires_grad=True)
    output = up_block(x, skip)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert skip.grad is not None
    for param in up_block.parameters():
        assert param.grad is not None


@pytest.mark.parametrize("res_block_type", [ResBlockBatchNorm, ResBlockGroupNorm])
def test_resblock_types(res_block_type):
    block = UpSampleDoubleResBlock(64, 32, res_block_type=res_block_type)
    assert isinstance(block.conv_blocks[0], res_block_type)
    assert isinstance(block.conv_blocks[1], res_block_type)


def test_custom_params():
    block = UpSampleDoubleResBlock(
        64, 32, res_block_type=ResBlockGroupNorm, res_block_params={"n_groups": 16}
    )
    assert block.conv_blocks[0].n_groups == 16
    assert block.conv_blocks[1].n_groups == 16


def test_skip_connection():
    block = UpSampleDoubleResBlock(64, 32)
    x = torch.randn(1, 64, 16, 16)
    skip = torch.randn(1, 32, 32, 32)
    output = block(x, skip)

    # Ensure the output has incorporated information from both inputs
    assert not torch.allclose(output, torch.zeros_like(output))
    assert output.shape == skip.shape


def test_forward_pass():
    for block_class in [UpSampleDoubleResBlock, SingleConvUpBlock]:
        block = block_class(64, 32)
        x = torch.randn(1, 64, 16, 16)
        skip = torch.randn(1, 32, 32, 32)
        output = block(x, skip)
        assert output.shape == (1, 32, 32, 32)
        assert torch.isfinite(output).all()
