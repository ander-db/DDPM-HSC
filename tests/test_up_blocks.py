import pytest
import torch
import torch.nn as nn
from src.blocks.Res_GroupNorm_SiLU_D import ResBlockGroupNorm
from src.blocks.Up_Res_GroupNorm_SiLU_D import (
    Up_Res_GroupNorm_SiLU_D,
)  # Adjust import path as needed


@pytest.fixture
def up_res_module():
    return Up_Res_GroupNorm_SiLU_D(in_channels=128, out_channels=64)


def test_up_res_module_initialization(up_res_module):
    assert isinstance(up_res_module, nn.Module)
    assert isinstance(up_res_module.up, nn.ConvTranspose2d)
    assert isinstance(up_res_module.module, nn.Sequential)
    assert len(up_res_module.module) == 2
    assert all(isinstance(layer, ResBlockGroupNorm) for layer in up_res_module.module)


def test_up_res_module_up_layer(up_res_module):
    assert up_res_module.up.in_channels == 128
    assert up_res_module.up.out_channels == 64
    assert up_res_module.up.kernel_size == (2, 2)
    assert up_res_module.up.stride == (2, 2)


def test_up_res_module_forward():
    module = Up_Res_GroupNorm_SiLU_D(in_channels=128, out_channels=64)
    g = torch.randn(1, 128, 16, 16)
    x = torch.randn(1, 64, 32, 32)

    output = module(x=x, g=g)

    assert output.shape == (1, 64, 32, 32)


def test_up_res_module_forward_different_sizes():
    module = Up_Res_GroupNorm_SiLU_D(in_channels=256, out_channels=128)
    g = torch.randn(2, 256, 8, 8)
    x = torch.randn(2, 128, 16, 16)

    output = module(x=x, g=g)

    assert output.shape == (2, 128, 16, 16)


def test_up_res_module_custom_args():
    module = Up_Res_GroupNorm_SiLU_D(
        in_channels=128, out_channels=64, n_groups=16, dropout_rate=0.2
    )

    assert isinstance(module.module[0], ResBlockGroupNorm)
    assert module.module[0].n_groups == 16
    assert module.module[0].dropout_rate == 0.2


def test_up_res_module_raises_error_on_mismatched_sizes():
    module = Up_Res_GroupNorm_SiLU_D(in_channels=128, out_channels=64)
    g = torch.randn(1, 128, 16, 16)
    x = torch.randn(1, 64, 64, 64)  # Mismatched size

    with pytest.raises(RuntimeError):
        module(x=x, g=g)


def test_up_res_module_output_type(up_res_module):
    g = torch.randn(1, 128, 16, 16)
    x = torch.randn(1, 64, 32, 32)
    output = up_res_module(x=x, g=g)
    assert isinstance(output, torch.Tensor)
    assert output.dtype == x.dtype
