from typing import List, Optional
import torch
import torch.nn as nn
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
        loss_fn=nn.MSELoss(),
        dropout_rate=0.1,
        lr=1e-4,
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

        self._build_network()

        self.ref = []
        self.preds = []
        self.targets = []

    def _build_network(self):
        self.first_layer = ResBlockGroupNorm(self.in_channels, self.encoder_channels[0])

        self.encoder = nn.ModuleList(
            [
                Down_Res_GroupNorm_SiLU_D(
                    self.encoder_channels[i],
                    self.encoder_channels[i + 1],
                    dropout_rate=self.dropout_rate,
                )
                for i in range(len(self.encoder_channels) - 2)
            ]
        )

        self.mid = Down_Res_GroupNorm_SiLU_D(
            self.encoder_channels[-2],
            self.encoder_channels[-1],
            dropout_rate=self.dropout_rate,
        )

        self.decoder = nn.ModuleList(
            [
                Up_Res_GroupNorm_SiLU_D(
                    self.decoder_channels[i],
                    self.decoder_channels[i + 1],
                    dropout_rate=self.dropout_rate,
                )
                for i in range(len(self.decoder_channels) - 1)
            ]
        )

        self.last_layer = ResBlockGroupNorm(
            self.decoder_channels[-1], self.out_channels, dropout_rate=self.dropout_rate
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
        self.time_embeddings = nn.ModuleList(
            [
                TimeEmbeddingProjectionLinearSiLU(self.time_embedding_dim, channel)
                for channel in self.encoder_channels
                + [self.encoder_channels[-1] * 2]
                + self.decoder_channels
            ]
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, None)
        loss = self.loss_fn(y_hat, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, None)
        loss = self.loss_fn(y_hat, y)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):

        x, y = batch
        prediction = self.forward(x)

        self.ref.append(x)
        self.preds.append(prediction)
        self.targets.append(y)

        return None

    def on_test_epoch_end(self) -> None:

        if not self.trainer.max_epochs:
            return

        if self.current_epoch < self.trainer.max_epochs- 1:
            return

        l1 = nn.L1Loss(reduction="none")
        l2 = nn.MSELoss(reduction="none")

        all_preds = torch.cat(self.preds)
        all_targets = torch.cat(self.targets)
        # all_refs = torch.cat(self.ref)

        # Calculate loss and std deviation
        l1_loss = l1(all_preds, all_targets).mean(dim=(1, 2, 3))
        print(f"[INFO] l1_loss.shape: {l1_loss.shape}")
        l1_loss_mean, l1_loss_std = l1_loss.mean(), l1_loss.std()

        l2_loss = l2(all_preds, all_targets).mean(dim=(1, 2, 3))
        print(f"[INFO] l2_loss.shape: {l2_loss.shape}")
        l2_loss_mean, l2_loss_std = l2_loss.mean(), l2_loss.std()

        self.log(
            "test/l1_mean", l1_loss_mean, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test/l1_std", l1_loss_std, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "test/l2_mean", l2_loss_mean, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test/l2_std", l2_loss_std, on_step=False, on_epoch=True, prog_bar=False
        )

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, eps=1e-4)
