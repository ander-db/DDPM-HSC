import torch
import torch.nn as nn
import lightning as L
from typing import Any
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio


class LogVisionMetricsBase(Callback):
    def __init__(self, batch_idx: int = -1, log_every_n_epochs: int = -1):
        super().__init__()

        self.batch_idx = batch_idx
        self.log_every_n_epochs = log_every_n_epochs

        assert self.batch_idx >= -1, "batch_idx should be greater or equal to -1"
        assert (
            self.log_every_n_epochs >= -1
        ), "log_every_n_epochs should be greater or equal to -1"

    @torch.no_grad()
    def calc_preds(self, pl_module: "L.LightningModule", ref: torch.Tensor):
        raise NotImplementedError

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.batch_idx != -1 and batch_idx != self.batch_idx:
            return

        preds = torch.cat(pl_module.train_preds)
        target = torch.cat(pl_module.train_targets)

        self._log_metrics(
            pl_module=pl_module, target=target, preds=preds, prefix="train"
        )

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

        if self.batch_idx != -1 and batch_idx != self.batch_idx:
            return

        preds = torch.cat(pl_module.val_preds)
        target = torch.cat(pl_module.val_targets)

        self._log_metrics(pl_module=pl_module, target=target, preds=preds, prefix="val")

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

        if self.batch_idx != -1 and batch_idx != self.batch_idx:
            return

        preds = torch.cat(pl_module.test_preds)
        target = torch.cat(pl_module.test_targets)

        self._log_metrics(
            pl_module=pl_module, target=target, preds=preds, prefix="test"
        )

    @torch.no_grad()
    def _calc_psnr(self, preds, targets):
        psnr = peak_signal_noise_ratio(
            preds,
            targets,
            reduction="none",
            data_range=(-1, 1),
            dim=(1, 2, 3),
        )

        mean_psnr = psnr.mean()
        std_psnr = psnr.std()

        return mean_psnr, std_psnr

    @torch.no_grad()
    def _calc_ssim(self, preds, target):
        ssim = structural_similarity_index_measure(
            preds, target, reduction="none", data_range=(-1, 1)
        )
        if isinstance(ssim, tuple):
            ssim = ssim[0]

        mean_ssim = ssim.mean()
        std_ssim = ssim.std()
        return mean_ssim, std_ssim

    @torch.no_grad()
    def _calc_mae(self, preds, target):
        _mae = nn.L1Loss(reduction="none")
        mae = _mae(preds, target).mean(dim=(1, 2, 3))

        mean_mae = mae.mean()
        std_mae = mae.std()

        return mean_mae, std_mae

    @torch.no_grad()
    def _calc_mse(self, preds, target):
        _mse = nn.MSELoss(reduction="none")
        mse = _mse(preds, target).mean(dim=(1, 2, 3))

        mean_mse = mse.mean()
        std_mse = mse.std()

        return mean_mse, std_mse

    @torch.no_grad()
    def _log_metrics(self, *, pl_module, target, preds, prefix):
        # Calculate metrics
        psnr, std_psnr = self._calc_psnr(preds, target)
        ssim, std_ssim = self._calc_ssim(preds, target)
        mae, std_mae = self._calc_mae(preds, target)
        mse, std_mse = self._calc_mse(preds, target)

        # PSNR
        pl_module.log(f"{prefix}/psnr", psnr)
        pl_module.log(f"{prefix}/psnr_std", std_psnr)

        # SSIM
        pl_module.log(f"{prefix}/ssim", ssim)
        pl_module.log(f"{prefix}/ssim_std", std_ssim)

        # MAE
        pl_module.log(f"{prefix}/mae", mae)
        pl_module.log(f"{prefix}/mae_std", std_mae)

        # MSE
        pl_module.log(f"{prefix}/mse", mse)
        pl_module.log(f"{prefix}/mse_std", std_mse)


class LogVisionMetricsDDPM(LogVisionMetricsBase):

    def __init__(self, batch_idx: int = 0, log_every_n_epochs: int = -1):
        super().__init__(batch_idx=batch_idx, log_every_n_epochs=log_every_n_epochs)

    @torch.no_grad()
    def calc_preds(self, pl_module: "L.LightningModule", ref: torch.Tensor):
        return pl_module.sample(ref)


    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.batch_idx != -1 and batch_idx != self.batch_idx:
            return

        preds = torch.cat(pl_module.train_preds)
        target = torch.cat(pl_module.train_targets)

        self._log_metrics(
            pl_module=pl_module, target=target, preds=preds, prefix="train"
        )

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

        if self.batch_idx != -1 and batch_idx != self.batch_idx:
            return

        preds = torch.cat(pl_module.val_preds)
        target = torch.cat(pl_module.val_targets)

        self._log_metrics(pl_module=pl_module, target=target, preds=preds, prefix="val")

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

        if self.batch_idx != -1 and batch_idx != self.batch_idx:
            return

        preds = torch.cat(pl_module.test_preds)
        target = torch.cat(pl_module.test_targets)

        self._log_metrics(
            pl_module=pl_module, target=target, preds=preds, prefix="test"
        )


class LogVisionMetricsUNet(LogVisionMetricsBase):

    def __init__(self, batch_idx: int = 0, log_every_n_epochs: int = -1):
        super().__init__(batch_idx=batch_idx, log_every_n_epochs=log_every_n_epochs)

    @torch.no_grad()
    def calc_preds(self, pl_module: "L.LightningModule", ref: torch.Tensor):
        return pl_module.forward(ref, t=None)
