import torch
import torch.nn as nn
from typing import List, Optional
from torch.optim.adam import Adam
import lightning as l

from src.blocks.Res_GroupNorm_SiLU_D import ResBlockGroupNorm
from src.blocks.Down_Res_GroupNorm_SiLU_D import Down_Res_GroupNorm_SiLU_D
from src.blocks.TE_projection_linear_SiLU import TimeEmbeddingProjectionLinearSiLU
from src.attention.spatial_attention import SpatialAttentionUNet
from src.blocks.Up_Res_GroupNorm_SiLU_D import Up_Res_GroupNorm_SiLU_D

"""UNet_Res_GroupNorm_SiLU_D is a class that defines a U-Net architecture with"""


class UNet_Res_GroupNorm_SiLU_D(l.LightningModule):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        encoder_channels: List[int] = [64, 128, 256, 512],
        time_embedding_dim: int | None = None,
        loss_fn: nn.Module = nn.L1Loss(),
        dropout_rate: float = 0.1,
        lr: float = 1e-4,
        n_groups: int = 32,
    ):
        super(UNet_Res_GroupNorm_SiLU_D, self).__init__()
        self.save_hyperparameters(ignore=["loss_fn"])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channels = encoder_channels
        self.decoder_channels = encoder_channels[::-1]
        self.time_embedding_dim = time_embedding_dim
        self.loss_fn = loss_fn
        self.dropout_rate = dropout_rate
        self.lr = lr

        self.n_groups = n_groups

        self._build_network()

        # Note: This is usefull to not repeat predictions in the callbacks
        # Training storage
        self.train_ref = []
        self.train_preds = []
        self.train_targets = []

        # Validation storage
        self.val_ref = []
        self.val_preds = []
        self.val_targets = []

        # Test storage
        self.test_ref = []
        self.test_preds = []
        self.test_targets = []

    def _build_network(self):
        self.first_layer = ResBlockGroupNorm(
            self.in_channels,
            self.encoder_channels[0],
            n_groups=min(self.in_channels, self.n_groups),
        )

        self.encoder = nn.ModuleList(
            [
                Down_Res_GroupNorm_SiLU_D(
                    self.encoder_channels[i],
                    self.encoder_channels[i + 1],
                    dropout_rate=self.dropout_rate,
                    n_groups=min(self.encoder_channels[i], self.n_groups),
                )
                for i in range(len(self.encoder_channels) - 2)
            ]
        )

        self.mid = Down_Res_GroupNorm_SiLU_D(
            self.encoder_channels[-2],
            self.encoder_channels[-1],
            dropout_rate=self.dropout_rate,
            n_groups=min(self.encoder_channels[-1], self.n_groups),
        )

        self.decoder = nn.ModuleList(
            [
                Up_Res_GroupNorm_SiLU_D(
                    self.decoder_channels[i],
                    self.decoder_channels[i + 1],
                    dropout_rate=self.dropout_rate,
                    n_groups=min(
                        self.decoder_channels[i + 1],
                        self.decoder_channels[i],
                        self.n_groups,
                    ),
                )
                for i in range(len(self.decoder_channels) - 1)
            ]
        )

        self.last_layer = ResBlockGroupNorm(
            self.decoder_channels[-1],
            self.out_channels,
            dropout_rate=self.dropout_rate,
            n_groups=min(self.decoder_channels[-1], self.out_channels, self.n_groups),
        )

        self.attention = nn.ModuleList(
            [
                SpatialAttentionUNet(
                    self.decoder_channels[i] * 2, dropout_rate=self.dropout_rate
                )
                for i in range(1, len(self.decoder_channels))
            ]
        )

        if self.time_embedding_dim is not None:
            self._build_time_embeddings()

    def _build_time_embeddings(self):

        self.time_embeddings = nn.ModuleList()

        for _channels in self.encoder_channels + self.decoder_channels[1:]:
            self.time_embeddings.append(
                TimeEmbeddingProjectionLinearSiLU(self.time_embedding_dim, _channels)
            )

    def forward(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return (
            self._forward_with_time_embeddings(x, t)
            if self.time_embedding_dim is not None
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
            t_emb = self.time_embeddings[i + 1](t)
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

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.train_ref = []
            self.train_preds = []
            self.train_targets = []

        ref, true = batch
        prediction = self(ref, None)
        loss = self.loss_fn(prediction, true)

        self.train_ref.append(ref)
        self.train_targets.append(true)
        self.train_preds.append(prediction)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.val_ref = []
            self.val_preds = []
            self.val_targets = []

        ref, true = batch
        prediction = self(ref, None)
        loss = self.loss_fn(prediction, true)

        self.val_ref.append(ref)
        self.val_targets.append(true)
        self.val_preds.append(prediction)

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        if batch_idx == 0:
            self.test_ref = []
            self.test_preds = []
            self.test_targets = []

        print("test_step batch_idx:", batch_idx)
        print('test_ref len:', len(self.test_ref))

        ref, true = batch
        prediction = self.forward(ref)

        self.test_ref.append(ref)
        self.test_targets.append(true)
        self.test_preds.append(prediction)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, eps=1e-4)
