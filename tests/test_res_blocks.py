import warnings
import pytest
import torch
import torch.nn as nn

from src.res_blocks import (
    ResBlock,
    ResBlockGroupNorm,
    ResBlockBatchNorm,
)


@pytest.fixture
def input_tensor():
    return torch.randn(1, 64, 32, 32)


def test_resblock_abstract():
    with pytest.raises(NotImplementedError):
        ResBlock(64, 64)


@pytest.mark.parametrize("block_class", [ResBlockGroupNorm, ResBlockBatchNorm])
@pytest.mark.parametrize("in_channels,out_channels", [(64, 64), (64, 128)])
def test_resblock_initialization(block_class, in_channels, out_channels):
    block = block_class(in_channels, out_channels)
    assert isinstance(block.main_path, nn.Sequential)
    assert block.in_channels == in_channels
    assert block.out_channels == out_channels
    if in_channels != out_channels:
        assert isinstance(block.residual_connection, nn.Conv2d)
    else:
        assert block.residual_connection is None


@pytest.mark.parametrize("block_class", [ResBlockGroupNorm, ResBlockBatchNorm])
@pytest.mark.parametrize("in_channels,out_channels", [(64, 64), (64, 128)])
def test_resblock_forward(block_class, in_channels, out_channels, input_tensor):
    block = block_class(in_channels, out_channels)
    output = block(input_tensor)
    assert output.shape == (1, out_channels, 32, 32)


def test_resblockgroupnorm_structure():
    block = ResBlockGroupNorm(64, 128, n_groups=32, dropout_rate=0.1)
    assert isinstance(block.main_path[0], nn.GroupNorm)
    assert isinstance(block.main_path[1], nn.SiLU)
    assert isinstance(block.main_path[2], nn.Conv2d)
    assert isinstance(block.main_path[3], nn.Dropout)
    assert block.main_path[2].padding_mode == "zeros"


def test_resblockbatchnorm_structure():
    block = ResBlockBatchNorm(64, 128, dropout_rate=0.1)
    assert isinstance(block.main_path[0], nn.BatchNorm2d)
    assert isinstance(block.main_path[1], nn.ReLU)
    assert isinstance(block.main_path[2], nn.Conv2d)
    assert isinstance(block.main_path[3], nn.Dropout)
    assert block.main_path[2].padding_mode == "zeros"


@pytest.mark.parametrize("block_class", [ResBlockGroupNorm, ResBlockBatchNorm])
def test_resblock_residual_connection(block_class, input_tensor):
    block = block_class(64, 64)
    output_with_input = block(input_tensor)
    output_without_input = block.main_path(input_tensor)
    assert not torch.allclose(output_with_input, output_without_input)


@pytest.mark.parametrize("block_class", [ResBlockGroupNorm, ResBlockBatchNorm])
def test_resblock_dropout(block_class):
    block = block_class(64, 64, dropout_rate=0.5)
    block.train()
    input_tensor = torch.randn(1, 64, 32, 32)
    output1 = block(input_tensor)
    output2 = block(input_tensor)
    assert not torch.allclose(output1, output2)

    block.eval()
    output1 = block(input_tensor)
    output2 = block(input_tensor)
    assert torch.allclose(output1, output2)


def test_resblockgroupnorm_groups():
    block = ResBlockGroupNorm(64, 64, n_groups=16)
    assert block.main_path[0].num_groups == 16
    assert block.main_path[4].num_groups == 16


@pytest.mark.parametrize("dropout_rate", [0, 0.1, 0.5])
def test_resblock_dropout_rate(dropout_rate):
    block_gn = ResBlockGroupNorm(64, 64, dropout_rate=dropout_rate)
    block_bn = ResBlockBatchNorm(64, 64, dropout_rate=dropout_rate)

    for block in [block_gn, block_bn]:
        if dropout_rate == 0:
            assert isinstance(block.main_path[3], nn.Identity)
            assert isinstance(block.main_path[-1], nn.Identity)
        else:
            assert isinstance(block.main_path[3], nn.Dropout)
            assert isinstance(block.main_path[-1], nn.Dropout)
            assert block.main_path[3].p == dropout_rate
            assert block.main_path[-1].p == dropout_rate


@pytest.mark.parametrize(
    "in_channels,out_channels,n_groups,expect_warning",
    [
        (64, 128, 256, True),  # More groups than both in_channels and out_channels
        (64, 32, 128, True),  # More groups than out_channels
        (32, 64, 128, True),  # More groups than in_channels
        (64, 128, 32, False),  # Normal case, no warnings
    ],
)
def test_resblockgroupnorm_group_adjustment(
    in_channels, out_channels, n_groups, expect_warning
):
    if expect_warning:
        with pytest.warns(UserWarning) as warning_info:
            block = ResBlockGroupNorm(in_channels, out_channels, n_groups=n_groups)
        assert len(warning_info) > 0, "Expected a warning, but none was raised"
        assert "Number of groups" in str(warning_info[0].message)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            block = ResBlockGroupNorm(in_channels, out_channels, n_groups=n_groups)

    # Verify that the GroupNorm layers have the correct number of groups
    assert block.main_path[0].num_groups == min(n_groups, in_channels)
    assert block.main_path[4].num_groups == min(n_groups, out_channels)

    # Test forward pass
    input_tensor = torch.randn(1, in_channels, 32, 32)
    output = block(input_tensor)
    assert output.shape == (1, out_channels, 32, 32)


def test_resblockgroupnorm_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        ResBlockGroupNorm(64, 128, n_groups=32)  # This should not raise any warnings
