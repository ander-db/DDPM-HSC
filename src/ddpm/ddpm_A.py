import torch
import torch.nn as nn
from torch.optim.adam import Adam

import lightning as l

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from typing import List, Tuple
from src.unet.UNet_Res_GroupNorm_SiLU_D import UNet_Res_GroupNorm_SiLU_D
from src.blocks.PositionalEncoding import PositionalEncoding


class DDPM_2D(l.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        diffusion_steps: int = 50,
        pe_emb_dim: int = 64,
        lr: float = 1e-3,
        dropout: float = 0.05,
        n_groups: int = 32,
        beta_start: float = 0.002,
        beta_end: float = 0.2,
        encoder_channels: List[int] = [64, 128, 256, 512],
        loss_fn: nn.Module = nn.MSELoss(),
        scheduler: str = "squaredcos_cap_v2",
    ):
        super(DDPM_2D, self).__init__()

        self.save_hyperparameters(ignore="loss_fn")
        self.diffusion_steps = diffusion_steps
        self.pe_emb_dim = pe_emb_dim
        self.lr = lr
        self.dropout = dropout
        self.groups = n_groups
        self.scheduler = scheduler
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channels = encoder_channels
        self.decoder_channels = encoder_channels[::-1]
        self.loss = loss_fn

        self.time_embedding = PositionalEncoding(pe_emb_dim, diffusion_steps)
        self.noise_scheduler = DDPMScheduler(
            diffusion_steps, beta_start, beta_end, scheduler
        )

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

    @torch.no_grad()
    def _build_network(self):
        self.model = UNet_Res_GroupNorm_SiLU_D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            encoder_channels=self.encoder_channels,
            time_embedding_dim=self.pe_emb_dim,
            loss_fn=self.loss,
            dropout_rate=self.dropout,
            lr=self.lr,
            n_groups=self.groups,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pe_emb = self.time_embedding(t.detach())

        return self.model(x, pe_emb)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, eps=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        if batch_idx == 0:
            self.train_ref = []
            self.train_preds = []
            self.train_targets = []

        ref, x = batch
        batch_size = x.shape[0]

        t = self._generate_random_timesteps(batch_size)

        noisy_batch, noise = self._add_noise(x, t)
        noisy_ref_concat = torch.cat(
            [noisy_batch, ref], dim=1
        )  # Concatenate the reference image with the noisy image to use it as guidance

        noise_prediction = self.forward(noisy_ref_concat, t)
        loss = self.loss(noise_prediction, noise).mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Compute the validation loss for the model by sampling the images and computing the loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Tuple containing the reference and input images
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Validation loss
        """
        if batch_idx == 0:
            self.val_ref = []
            self.val_preds = []
            self.val_targets = []

        ref, x = batch

        t = self._generate_random_timesteps(x.shape[0])
        noisy_batch, noise = self._add_noise(x, t)
        noisy_ref_concat = torch.cat([noisy_batch, ref], dim=1)

        loss = self.loss(self.forward(noisy_ref_concat, t), noise).mean()
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Compute the test loss for the model by sampling the images and computing the loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Tuple containing the reference and input images
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Test loss
        """

        if batch_idx == 0:
            self.test_ref = []
            self.test_preds = []
            self.test_targets = []

        ref, x = batch
        predicted_samples = self.sample(ref)

        self.test_ref.append(ref)
        self.test_preds.append(predicted_samples)
        self.test_targets.append(x)

    @torch.no_grad()
    def _add_noise(
        self, x: torch.FloatTensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to the input image x at timestep t

        Args:
            x (torch.FloatTensor): Input image
            t (torch.IntTensor): Timestep

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Noisy image and noise tensor

        """
        noise = torch.randn_like(x).to(x.dtype)  # Ensure noise matches x's dtype

        noise_img = self.noise_scheduler.add_noise(x, noise, t)
        return noise_img, noise

    @torch.no_grad()
    def _generate_random_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.diffusion_steps, (batch_size,))

    @torch.no_grad()
    def sample(self, ref: torch.Tensor) -> torch.Tensor:
        """
        Sample from the model

        Args:
            x (torch.FloatTensor): Input image batch
            ref (torch.FloatTensor): Reference image batch

        Returns:
            torch.FloatTensor: Sampled image
        """

        device = self.device
        x = torch.randn_like(ref, device=device)
        self.noise_scheduler.set_timesteps(self.diffusion_steps, device=device)

        # Repite for each timestep
        for i in range(self.diffusion_steps - 1, -1, -1):
            model_input = torch.cat([x, ref], dim=1)

            t = torch.full((ref.shape[0],), i, device=device, dtype=torch.long)
            predicted_noise = self.forward(model_input, t)

            x = self.noise_scheduler.step(
                model_output=predicted_noise,
                timestep=i,
                sample=x[:, :1],
                return_dict=False,
            )[0]

        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Output must be a float tensor")

        return x
