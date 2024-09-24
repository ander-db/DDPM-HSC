import torch
import torch.nn as nn

import lightning as l

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


from typing import List
from src.unet.UNet_Res_GroupNorm_SiLU_D import UNet_Res_GroupNorm_SiLU_D
from src.blocks.PositionalEncoding import PositionalEncoding


class DDPM_A(l.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        diffusion_steps: int = 1000,
        pe_emb_dim=64,
        lr: float = 1e-3,
        dropout: float = 0.05,
        groups: int = 32,
        beta_start: float = 0.002,
        beta_end: float = 0.2,
        encoder_channels: List[int] = [64, 128, 256, 512],
        scheduler: str = "squaredcos_cap_v2",
    ):
        super(DDPM_A, self).__init__()

        self.save_hyperparameters()
        self.diffusion_steps = diffusion_steps
        self.pe_emb_dim = pe_emb_dim
        self.lr = lr
        self.dropout = dropout
        self.groups = groups
        self.scheduler = scheduler
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channels = encoder_channels
        self.decoder_channels = encoder_channels[::-1]

        self.time_embedding = PositionalEncoding(pe_emb_dim, diffusion_steps)
        self.noise_scheduler = DDPMScheduler(
            diffusion_steps, beta_start, beta_end, scheduler
        )

        self._build_network()

    def _build_network(self):
        self.unet = UNet_Res_GroupNorm_SiLU_D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            encoder_channels=self.encoder_channels,
            loss_fn=nn.MSELoss(),
        )


    def forward(self, x, t):
        with torch.no_grad():
            pe_emb = self.time_embedding(t.detach())

        return self.model(x, pe_emb)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-4)
        return optimizer
