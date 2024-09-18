import torch
import torch.nn as nn

import lightning as l

from src.blocks.Res_GroupNorm_SiLU_D import ResBlockGroupNorm
from src.blocks.Up_Res_BatchNorm_ReLU_D import Up_Res_GroupNorm_SiLU_D
from src.blocks.Down_Res_GroupNorm_SiLU_D import (
    Down_Res_GroupNorm_SiLU_D,
)
from src.attention.spatial_attention import SpatialAttentionUNet


class UNet_Res_GroupNorm_SiLU_D(l.LightningModule):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        encoder_channel_layers: list[int] = [64, 128, 256, 512],
        decoder_channel_layers: list[int] = [512, 256, 128, 64],
    ):

        super().__init__()

        self.first_layer = ResBlockGroupNorm(in_channels, encoder_channel_layers[0])

        self.encoder = [
            Down_Res_GroupNorm_SiLU_D(
                encoder_channel_layers[i], encoder_channel_layers[i + 1]
            )
            for i in range(len(encoder_channel_layers) - 1)
        ]

        self.mid = ResBlockGroupNorm(
            encoder_channel_layers[-1], decoder_channel_layers[0]
        )
        self.decoder = [
            Up_Res_GroupNorm_SiLU_D(
                decoder_channel_layers[i], decoder_channel_layers[i + 1]
            )
            for i in range(len(decoder_channel_layers) - 1)
        ]

        self.attention = nn.ModuleList(
            [
                SpatialAttentionUNet(decoder_channel_layers[i])
                for i in range(len(decoder_channel_layers))
            ]
        )

        self.last_layer = ResBlockGroupNorm(decoder_channel_layers[-1], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)

        skip_connections = [x]

        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            skip_connections.append(x)

        x = self.mid(x)

        for i in range(len(self.decoder)):
            attention = self.attention[i](x, skip_connections[-(i + 2)])
            x = self.decoder[i](x, attention)

        x = self.last_layer(x)

        return x
