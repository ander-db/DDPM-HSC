import torch
import lightning as L
from typing import Any
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio


class LogVisionMetricsBase(Callback):
    """
    Callback to log vision metrics (SSIM, PSNR) for the denoising model.

    Args:
    - log_every_n_epochs (int): Log the metrics every n epochs
    - batch_idx (int): Index of the batch to log. If batch_idx is -1, log the metrics for all batches
    """

    def __init__(self, batch_idx: int = 0, log_every_n_epochs: int = -1):
        super().__init__()

        self.batch_idx = batch_idx
        self.log_every_n_epochs = log_every_n_epochs

        assert self.batch_idx >= -1, "batch_idx should be greater or equal to -1"
        assert (
            self.log_every_n_epochs >= -1
        ), "log_every_n_epochs should be greater or equal to -1"

    @torch.no_grad()
    def calc_preds(self, pl_module: "L.LightningModule", ref: torch.Tensor):
        pass

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.calc_batch_metrics(pl_module, outputs, batch, batch_idx, prefix="train")

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.calc_batch_metrics(pl_module, outputs, batch, batch_idx, prefix="val")

    @torch.no_grad()
    def on_test_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.calc_batch_metrics(pl_module, outputs, batch, batch_idx, prefix="test")

    @torch.no_grad()
    def calc_batch_metrics(self, pl_module, outputs, batch, batch_idx, prefix):
        if self.batch_idx != -1 and batch_idx != self.batch_idx:
            return

        if (
            self.log_every_n_epochs != -1
            and pl_module.current_epoch % self.log_every_n_epochs != 0
        ):
            return

        ref, x = batch
        ref = ref.to(pl_module.device)
        x = x.to(pl_module.device)

        # Log metrics
        preds = self.calc_preds(pl_module=pl_module, ref=ref)

        self._log_metrics(pl_module=pl_module, target=x, preds=preds, prefix=prefix)

    @torch.no_grad()
    def _calc_psnr(self, preds, targets):
        return peak_signal_noise_ratio(preds, targets)

    @torch.no_grad()
    def _calc_ssim(self, preds, target):
        ssim = structural_similarity_index_measure(preds, target)
        if isinstance(ssim, tuple):
            ssim = ssim[0]
        return ssim

    @torch.no_grad()
    def _calc_mae(self, preds, target):
        return torch.mean(torch.abs(preds - target))

    @torch.no_grad()
    def _log_metrics(self, *, pl_module, target, preds, prefix):
        psnr = self._calc_psnr(preds, target)
        ssim = self._calc_ssim(preds, target)
        mae = self._calc_mae(preds, target)

        pl_module.log(f"{prefix}/psnr", psnr)
        pl_module.log(f"{prefix}/ssim", ssim)
        pl_module.log(f"{prefix}/mae", mae)


class LogVisionMetricsDDPM(LogVisionMetricsBase):
    """
    Callback to log vision metrics (SSIM, PSNR, LPIPS) for the DDPM denoising model.

    Args:
    - log_every_n_epochs (int): Log the metrics every n epochs
    - batch_idx (int): Index of the batch to log. If batch_idx is -1, log the metrics for all batches
    """

    def __init__(self, batch_idx: int = 0, log_every_n_epochs: int = -1):
        super().__init__(batch_idx=batch_idx, log_every_n_epochs=log_every_n_epochs)

    @torch.no_grad()
    def calc_preds(self, pl_module: "L.LightningModule", ref: torch.Tensor):
        return pl_module.sample(ref)


class LogVisionMetricsUNet(LogVisionMetricsBase):
    """
    Callback to log vision metrics (SSIM, PSNR, LPIPS) for the DDPM denoising model.

    Args:
    - log_every_n_epochs (int): Log the metrics every n epochs
    - batch_idx (int): Index of the batch to log. If batch_idx is -1, log the metrics for all batches
    """

    def __init__(self, batch_idx: int = 0, log_every_n_epochs: int = -1):
        super().__init__(batch_idx=batch_idx, log_every_n_epochs=log_every_n_epochs)

    @torch.no_grad()
    def calc_preds(self, pl_module: "L.LightningModule", ref: torch.Tensor):
        return pl_module.forward(ref, t=None)
