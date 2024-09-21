import pytest
import torch
from torch import nn

# Asumiendo que la clase SpatialAttentionUNet está definida en un archivo llamado model.py
from src.attention.spatial_attention import SpatialAttentionUNet


@pytest.fixture
def attention_block():
    return SpatialAttentionUNet(g_channels=64)


@pytest.fixture
def input_tensors():
    batch_size = 2
    g_channels = 64
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
    ), f"La forma de la salida {output.shape} no coincide con la esperada {expected_output_shape}"
    assert (
        attention_map.shape == expected_attention_map_shape
    ), f"La forma del mapa de atención {attention_map.shape} no coincide con la esperada {expected_attention_map_shape}"


def test_spatial_attention_unet_output_type(attention_block, input_tensors):
    x, g = input_tensors
    output, attention_map = attention_block(g, x)

    assert isinstance(
        output, torch.Tensor
    ), "La salida debería ser un tensor de PyTorch"
    assert isinstance(
        attention_map, torch.Tensor
    ), "El mapa de atención debería ser un tensor de PyTorch"
    assert (
        output.dtype == x.dtype
    ), f"El tipo de datos de la salida {output.dtype} no coincide con el de la entrada {x.dtype}"
    assert (
        attention_map.dtype == x.dtype
    ), f"El tipo de datos del mapa de atención {attention_map.dtype} no coincide con el de la entrada {x.dtype}"


def test_spatial_attention_unet_gradient_flow(attention_block, input_tensors):
    x, g = input_tensors
    x.requires_grad = True
    g.requires_grad = True

    output, attention_map = attention_block(g, x)
    loss = output.sum() + attention_map.sum()
    loss.backward()

    assert x.grad is not None, "No hay gradiente fluyendo hacia x"
    assert g.grad is not None, "No hay gradiente fluyendo hacia g"


def test_attention_map_range(attention_block, input_tensors):
    x, g = input_tensors
    _, attention_map = attention_block(g, x)

    assert torch.all(attention_map >= 0) and torch.all(
        attention_map <= 1
    ), "Todos los valores del mapa de atención deben estar entre 0 y 1"
