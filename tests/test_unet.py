import pytest
import torch
import torch.nn as nn
from src.unet.UNet_Res_GroupNorm_SiLU_D import UNet_Res_GroupNorm_SiLU_D
from src.blocks.PositionalEncoding import PositionalEncoding


@pytest.fixture
def unet_model():
    return UNet_Res_GroupNorm_SiLU_D(
        in_channels=3,
        out_channels=3,
        encoder_channels=[64, 128, 256, 512],
        time_embedding_dim=None,
        loss_fn=nn.MSELoss(),
        lr=1e-4,
    )


@pytest.fixture
def unet_model_with_time():
    return UNet_Res_GroupNorm_SiLU_D(
        in_channels=3,
        out_channels=3,
        encoder_channels=[64, 128, 256, 512],
        time_embedding_dim=256,
        loss_fn=nn.MSELoss(),
        lr=1e-4,
    )


@pytest.fixture
def positional_encoding():
    return PositionalEncoding(embedding_dim=256, max_sequence_length=1000)


def test_unet_initialization(unet_model):
    assert isinstance(unet_model, UNet_Res_GroupNorm_SiLU_D)
    assert unet_model.in_channels == 3
    assert unet_model.out_channels == 3
    assert unet_model.encoder_channels == [64, 128, 256, 512]
    assert unet_model.decoder_channels == [512, 256, 128, 64]
    assert unet_model.time_embedding_dim is None
    assert isinstance(unet_model.loss_fn, nn.MSELoss)
    assert unet_model.lr == 1e-4


def test_unet_forward_pass(unet_model):
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 256, 256)
    output = unet_model(input_tensor, None)
    print(f"{output.shape=}")
    assert output.shape == (batch_size, 3, 256, 256)


def test_unet_with_time_embeddings(unet_model_with_time):
    assert unet_model_with_time.time_embedding_dim == 256
    assert (
        len(unet_model_with_time.time_embeddings)
        == len(unet_model_with_time.encoder_channels)
        + len(unet_model_with_time.decoder_channels)
        + 1
    )


def test_unet_forward_pass_with_time_embeddings(
    unet_model_with_time, positional_encoding
):
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 256, 256)
    time_steps = torch.randint(0, 1000, (batch_size,))
    time_embeddings = positional_encoding(time_steps)
    output = unet_model_with_time(input_tensor, time_embeddings)
    assert output.shape == (batch_size, 3, 256, 256)


def test_unet_loss_calculation(unet_model_with_time, positional_encoding):
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 256, 256)
    target_tensor = torch.randn(batch_size, 3, 256, 256)
    time_steps = torch.randint(0, 1000, (batch_size,))
    time_embeddings = positional_encoding(time_steps)
    output = unet_model_with_time(input_tensor, time_embeddings)
    loss = unet_model_with_time.loss_fn(output, target_tensor)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])


def test_unet_optimizer_configuration(unet_model_with_time):
    optimizer = unet_model_with_time.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == unet_model_with_time.lr


def test_unet_encoder_decoder_structure(unet_model_with_time):
    assert (
        len(unet_model_with_time.encoder)
        == len(unet_model_with_time.encoder_channels) - 1
    )
    assert len(unet_model_with_time.decoder) == len(
        unet_model_with_time.decoder_channels
    )
    assert len(unet_model_with_time.attention) == len(
        unet_model_with_time.decoder_channels
    )


@pytest.mark.parametrize(
    "batch_size, height, width",
    [
        (6, 64, 64),
        (4, 128, 128),
        (2, 256, 256),
    ],
)
def test_unet_different_input_sizes(
    unet_model_with_time, positional_encoding, batch_size, height, width
):
    input_tensor = torch.randn(batch_size, 3, height, width)
    time_steps = torch.randint(0, 1000, (batch_size,))
    time_embeddings = positional_encoding(time_steps)
    output = unet_model_with_time(input_tensor, time_embeddings)
    assert output.shape == (batch_size, 3, height, width)
