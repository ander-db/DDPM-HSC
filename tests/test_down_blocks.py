import pytest
import torch
import torch.nn as nn
from src.blocks.Down_Res_GroupNorm_SiLU_D import Down_Res_GroupNorm_SiLU_D
from src.blocks.Res_GroupNorm_SiLU_D import ResBlockGroupNorm


@pytest.fixture
def down_res_module():
    return Down_Res_GroupNorm_SiLU_D(in_channels=64, out_channels=128)


def test_down_res_module_initialization(down_res_module):
    assert isinstance(down_res_module, nn.Module)
    assert down_res_module.in_channels == 64
    assert down_res_module.out_channels == 128
    assert down_res_module.n_groups == 32
    assert down_res_module.dropout_rate == 0.1


def test_down_res_module_forward(down_res_module):
    # Create a random input tensor
    x = torch.randn(1, 64, 32, 32)

    # Pass the input through the module
    output = down_res_module(x)

    # Check the output shape
    assert output.shape == (1, 128, 16, 16)


def test_down_res_module_custom_params():
    custom_module = Down_Res_GroupNorm_SiLU_D(
        in_channels=32, out_channels=64, n_groups=16, dropout_rate=0.2
    )
    assert custom_module.in_channels == 32
    assert custom_module.out_channels == 64
    assert custom_module.n_groups == 16
    assert custom_module.dropout_rate == 0.2


def test_down_res_module_structure(down_res_module):
    assert len(down_res_module.module) == 3
    assert isinstance(down_res_module.module[0], nn.MaxPool2d)
    assert isinstance(down_res_module.module[1], ResBlockGroupNorm)
    assert isinstance(down_res_module.module[2], ResBlockGroupNorm)


def test_down_res_module_output_type(down_res_module):
    x = torch.randn(1, 64, 32, 32)
    output = down_res_module(x)
    assert isinstance(output, torch.Tensor)
    assert output.dtype == x.dtype
