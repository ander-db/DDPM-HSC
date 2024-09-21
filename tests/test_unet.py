import pytest
import torch
import torch.nn as nn
from src.unet.UNet_Res_GroupNorm_SiLU_D import UNet_Res_GroupNorm_SiLU_D


@pytest.fixture
def unet_model():
    return UNet_Res_GroupNorm_SiLU_D(
        in_channels=3,
        out_channels=3,
        encoder_channel_layers=[64, 128, 256, 512],
        time_embeddings_dim=None,
        loss_fn=nn.MSELoss(),
        lr=1e-4,
    )



@pytest.fixture
def unet_model_with_time():
   return UNet_Res_GroupNorm_SiLU_D(
       in_channels=3,
       out_channels=3,
       encoder_channel_layers=[64, 128, 256, 512],
       time_embeddings_dim=256,
       loss_fn=nn.MSELoss(),
       lr=1e-4
   )

def test_unet_initialization(unet_model):
   assert isinstance(unet_model, UNet_Res_GroupNorm_SiLU_D)
   assert unet_model.in_channels == 3
   assert unet_model.out_channels == 3
   assert unet_model.encoder_channel_layers == [64, 128, 256, 512]
   assert unet_model.decoder_channel_layers == [512, 256, 128, 64]
   assert unet_model.time_embeddings_dim is None
   assert isinstance(unet_model.loss_fn, nn.MSELoss)
   assert unet_model.lr == 1e-4

def test_unet_forward_pass(unet_model):
   batch_size = 4
   input_tensor = torch.randn(batch_size, 3, 256, 256)
   output = unet_model(input_tensor)
   assert output.shape == (batch_size, 3, 256, 256)

# def test_unet_with_time_embeddings(unet_model_with_time):
#    assert unet_model_with_time.time_embeddings_dim == 256
#    assert len(unet_model_with_time.time_embeddings) == len(unet_model_with_time.encoder_channel_layers) + len(unet_model_with_time.decoder_channel_layers)
#
# def test_unet_forward_pass_with_time_embeddings(unet_model_with_time):
#    batch_size = 4
#    input_tensor = torch.randn(batch_size, 3, 256, 256)
#    output = unet_model_with_time(input_tensor)
#    assert output.shape == (batch_size, 3, 256, 256)
#
# def test_unet_loss_calculation(unet_model):
#    batch_size = 4
#    input_tensor = torch.randn(batch_size, 3, 256, 256)
#    target_tensor = torch.randn(batch_size, 3, 256, 256)
#    output = unet_model(input_tensor)
#    loss = unet_model.loss_fn(output, target_tensor)
#    assert isinstance(loss, torch.Tensor)
#    assert loss.shape == torch.Size([])
#
# def test_unet_optimizer_configuration(unet_model):
#    optimizer = unet_model.configure_optimizers()
#    assert isinstance(optimizer, torch.optim.Adam)
#    assert optimizer.defaults['lr'] == unet_model.lr
#
# def test_unet_encoder_decoder_structure(unet_model):
#    assert len(unet_model.encoder) == len(unet_model.encoder_channel_layers) - 1
#    assert len(unet_model.decoder) == len(unet_model.decoder_channel_layers) - 1
#    assert len(unet_model.attention) == len(unet_model.decoder_channel_layers)
#
# @pytest.mark.parametrize("batch_size, height, width", [
#    (1, 128, 128),
#    (4, 256, 256),
#    (8, 512, 512),
# ])
# def test_unet_different_input_sizes(unet_model, batch_size, height, width):
#    input_tensor = torch.randn(batch_size, 3, height, width)
#    output = unet_model(input_tensor)
#    assert output.shape == (batch_size, 3, height, width)
