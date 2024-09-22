from typing import Optional
import torch
import torch.nn as nn
import lightning as l

from src.blocks.Res_GroupNorm_SiLU_D import ResBlockGroupNorm
from src.blocks.Down_Res_GroupNorm_SiLU_D import Down_Res_GroupNorm_SiLU_D
from src.blocks.TE_projection_linear_SiLU import TimeEmbeddingProjectionLinearSiLU
from src.attention.spatial_attention import SpatialAttentionUNet
from src.blocks.Up_Res_GroupNorm_SiLU_D import Up_Res_GroupNorm_SiLU_D


class UNet_Res_GroupNorm_SiLU_D(l.LightningModule):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        encoder_channel_layers=[64, 128, 256, 512],
        time_embeddings_dim=None,
        loss_fn=nn.MSELoss(),
        lr=1e-4,
    ):
        super(UNet_Res_GroupNorm_SiLU_D, self).__init__()
        self.save_hyperparameters(ignore=["loss_fn"])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channel_layers = encoder_channel_layers
        self.decoder_channel_layers = encoder_channel_layers[::-1]
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

        self.mid = Down_Res_GroupNorm_SiLU_D(
            self.encoder_channel_layers[-1], self.encoder_channel_layers[-1] * 2
        )

        self.decoder = nn.ModuleList(
            [
                Up_Res_GroupNorm_SiLU_D(
                    self.decoder_channel_layers[i] * 2,
                    self.decoder_channel_layers[i],
                )
                for i in range(len(self.decoder_channel_layers))
            ]
        )

        self.last_layer = ResBlockGroupNorm(
            self.decoder_channel_layers[-1], self.out_channels
        )

        self.attention = nn.ModuleList(
            [
                SpatialAttentionUNet(self.decoder_channel_layers[i] * 2)
                for i in range(len(self.decoder_channel_layers))
            ]
        )

        if self.time_embeddings_dim is not None:
            self._build_time_embeddings()

    def _build_time_embeddings(self):
        self.time_embeddings = nn.ModuleList(
            [
                TimeEmbeddingProjectionLinearSiLU(self.time_embeddings_dim, channel)
                for channel in self.encoder_channel_layers
                + [self.encoder_channel_layers[-1] * 2]
                + self.decoder_channel_layers
            ]
        )

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor]) -> torch.Tensor:
        return (
            self._forward_with_time_embeddings(x, t)
            if self.time_embeddings_dim is not None
            else self._forward_without_time_embeddings(x)
        )

    def _forward_without_time_embeddings(self, g):
        g = self.first_layer(g)
        skip_connections = [g]

        for encoder_layer in self.encoder:
            g = encoder_layer(g)
            skip_connections.append(g)

        g = self.mid(g)

        for i, decoder_layer in enumerate(self.decoder):
            attn, attn_map = self.attention[i](g=g, x=skip_connections[-(i + 1)])
            g = decoder_layer(g=g, x=attn)

        return self.last_layer(g)

    def _forward_with_time_embeddings(self, g, t):
        g = self.first_layer(g)
        t_emb = self.time_embeddings[0](t)
        g += t_emb[:, :, None, None]
        skip_connections = [g]

        for i, encoder_layer in enumerate(self.encoder):
            g = encoder_layer(g)
            t_emb = self.time_embeddings[len(skip_connections)](t)
            g += t_emb[:, :, None, None]
            skip_connections.append(g)

        g = self.mid(g)
        t_emb = self.time_embeddings[len(skip_connections)](t)
        g += t_emb[:, :, None, None]

        for i, decoder_layer in enumerate(self.decoder):
            attn, attn_map = self.attention[i](g=g, x=skip_connections[-(i + 1)])
            g = decoder_layer(g=g, x=attn)
            t_emb = self.time_embeddings[len(skip_connections) + i + 1](t)
            g += t_emb[:, :, None, None]

        return self.last_layer(g)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
