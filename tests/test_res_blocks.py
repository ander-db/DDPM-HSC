import pytest
import torch
import torch.nn as nn
from typing import Tuple, List

# Asumiendo que las clases están definidas en un archivo llamado models.py
from src.blocks.Res_BatchNorm_ReLU_D import ResBlockBatchNorm
from src.blocks.Res_GroupNorm_SiLU_D import ResBlockGroupNorm


@pytest.fixture(params=[ResBlockBatchNorm, ResBlockGroupNorm])
def res_block(request):
    return request.param


@pytest.fixture(params=[(32, 64)])
def channels(request) -> Tuple[int, int]:
    return request.param


@pytest.fixture
def input_tensor() -> torch.Tensor:
    return torch.randn(2, 32, 28, 28)


def test_resblock_output_shape(res_block, channels, input_tensor):
    in_channels, out_channels = channels
    block = res_block(in_channels, out_channels)
    output = block(input_tensor)

    expected_shape = (
        input_tensor.shape[0],
        out_channels,
        input_tensor.shape[2],
        input_tensor.shape[3],
    )
    assert (
        output.shape == expected_shape
    ), f"Output shape {output.shape} does not match expected shape {expected_shape}"


def test_resblock_residual_connection(res_block, channels, input_tensor):
    in_channels, out_channels = channels
    block = res_block(in_channels, out_channels)

    # Comprobamos si se ha creado una conexión residual
    has_residual = in_channels != out_channels
    assert (
        block.residual_connection is not None
    ) == has_residual, f"Residual connection should {'exist' if has_residual else 'not exist'} when in_channels={in_channels} and out_channels={out_channels}"


def test_resblock_forward_pass(res_block, channels, input_tensor):
    in_channels, out_channels = channels
    block = res_block(in_channels, out_channels)

    output = block(input_tensor)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"


def test_resblock_main_path_layers(res_block, channels):
    in_channels, out_channels = channels
    block = res_block(in_channels, out_channels)

    main_path_layers = list(block.main_path.children())
    expected_layer_types = [
        nn.BatchNorm2d if isinstance(block, ResBlockBatchNorm) else nn.GroupNorm,
        nn.ReLU if isinstance(block, ResBlockBatchNorm) else nn.SiLU,
        nn.Conv2d,
        nn.Dropout,
        nn.BatchNorm2d if isinstance(block, ResBlockBatchNorm) else nn.GroupNorm,
        nn.ReLU if isinstance(block, ResBlockBatchNorm) else nn.SiLU,
        nn.Conv2d,
        nn.Dropout,
    ]

    for layer, expected_type in zip(main_path_layers, expected_layer_types):
        assert isinstance(
            layer, expected_type
        ), f"Expected {expected_type}, but got {type(layer)}"


def test_resblock_group_norm_warning(channels):
    in_channels, out_channels = channels
    with pytest.warns(
        UserWarning, match="Number of groups .* is greater than the number of channels"
    ):
        ResBlockGroupNorm(in_channels, out_channels, n_groups=in_channels + 1)
