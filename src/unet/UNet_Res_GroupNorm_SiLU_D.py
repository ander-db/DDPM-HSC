import torch
import torch.nn as nn
import lightning as l

from src.blocks.Res_GroupNorm_SiLU_D import ResBlockGroupNorm
from src.blocks.Up_Res_BatchNorm_ReLU_D import Up_Res_GroupNorm_SiLU_D
from src.blocks.Down_Res_GroupNorm_SiLU_D import Down_Res_GroupNorm_SiLU_D
from src.blocks.TE_projection_linear_SiLU import TimeEmbeddingProjectionLinearSiLU
from src.attention.spatial_attention import SpatialAttentionUNet


class UNet_Res_GroupNorm_SiLU_D(l.LightningModule):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        encoder_channel_layers=[64, 128, 256, 512],
        decoder_channel_layers=[512, 256, 128, 64],
        time_embeddings_dim=None,
        loss_fn=nn.MSELoss(),
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channel_layers = encoder_channel_layers
        self.decoder_channel_layers = decoder_channel_layers
        self.time_embeddings_dim = time_embeddings_dim
        self.loss_fn = loss_fn
        self.lr = lr

        self._build_network()

    def _build_network(self):
        self.first_layer = ResBlockGroupNorm(
            self.in_channels, self.encoder_channel_layers[0]
        )

        self.encoder = nn.ModuleList(
            [
                Down_Res_GroupNorm_SiLU_D(
                    self.encoder_channel_layers[i], self.encoder_channel_layers[i + 1]
                )
                for i in range(len(self.encoder_channel_layers) - 1)
            ]
        )

        self.mid = ResBlockGroupNorm(
            self.encoder_channel_layers[-1], self.decoder_channel_layers[0]
        )

        self.decoder = nn.ModuleList(
            [
                Up_Res_GroupNorm_SiLU_D(
                    self.decoder_channel_layers[i], self.decoder_channel_layers[i + 1]
                )
                for i in range(len(self.decoder_channel_layers) - 1)
            ]
        )

        self.attention = nn.ModuleList(
            [
                SpatialAttentionUNet(self.decoder_channel_layers[i])
                for i in range(len(self.decoder_channel_layers))
            ]
        )

        self.last_layer = ResBlockGroupNorm(
            self.decoder_channel_layers[-1], self.out_channels
        )

        if self.time_embeddings_dim is not None:
            self._build_time_embeddings()

    def _build_time_embeddings(self):
        self.time_embeddings = nn.ModuleList(
            [
                TimeEmbeddingProjectionLinearSiLU(self.time_embeddings_dim, channel)
                for channel in self.encoder_channel_layers + self.decoder_channel_layers
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self._forward_with_time_embeddings(x)
            if self.time_embeddings_dim is not None
            else self._forward_without_time_embeddings(x)
        )

    def _forward_without_time_embeddings(self, x):
        x = self.first_layer(x)
        skip_connections = [x]

        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_connections.append(x)

        x = self.mid(x)

        for i, decoder_layer in enumerate(self.decoder):
            attention = self.attention[i](x, skip_connections[-(i + 2)])
            x = decoder_layer(x, attention)

        return self.last_layer(x)

    def _forward_with_time_embeddings(self, x):
        x = self.first_layer(x)
        x += self.time_embeddings[0](x)
        skip_connections = [x]

        for i, encoder_layer in enumerate(self.encoder):
            x = encoder_layer(x)
            x += self.time_embeddings[i + 1](x)
            skip_connections.append(x)

        x = self.mid(x)
        x += self.time_embeddings[len(self.encoder) + 1](x)

        for i, decoder_layer in enumerate(self.decoder):
            attention = self.attention[i](x, skip_connections[-(i + 2)])
            x = decoder_layer(x, attention)
            x += self.time_embeddings[len(self.encoder) + 2 + i](x)

        return self.last_layer(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
