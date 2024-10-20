from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import Callback
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio

__constants__ = ["LogVisionMetricsBase"]


class MetricsCallbackUNet(Callback):

    def __init__(self):
        super().__init__()

    def on_train_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        preds = torch.cat(pl_module.train_preds)
        target = torch.cat(pl_module.train_targets)

        log_metrics(pl_module=pl_module, target=target, preds=preds, prefix="train")

    def on_validation_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        preds = torch.cat(pl_module.val_preds)
        target = torch.cat(pl_module.val_targets)

        log_metrics(pl_module=pl_module, target=target, preds=preds, prefix="val")

    def on_test_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        preds = torch.cat(pl_module.test_preds)
        target = torch.cat(pl_module.test_targets)

        log_metrics(pl_module=pl_module, target=target, preds=preds, prefix="test")


class MetricsCallbackDDPM(Callback):

    def __init__(self, log_every_n_epochs: int = 10):
        super().__init__()

        self.log_every_n_epochs = log_every_n_epochs

        self.train_ref = []
        self.train_preds = []
        self.train_targets = []

        self.val_ref = []
        self.val_preds = []
        self.val_targets = []

        self.test_ref = []
        self.test_preds = []
        self.test_targets = []

    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:

        log_this_epoch = self.check_if_log_every_n_epochs(trainer)

        if not log_this_epoch:
            return

        if (
            self.log_every_n_epochs != -1
            and trainer.current_epoch % self.log_every_n_epochs != 0
        ):
            return

        if batch_idx == 0:
            self.train_ref = []
            self.train_preds = []
            self.train_targets = []

        ref, x = batch
        predicted_samples = pl_module.sample(ref)

        self.train_ref.append(ref)
        self.train_preds.append(predicted_samples)
        self.train_targets.append(x)

    @torch.no_grad()
    def check_if_log_every_n_epochs(self, trainer: "L.Trainer") -> bool:
        return (
            self.log_every_n_epochs != -1
            and trainer.current_epoch % self.log_every_n_epochs == 0
        )

    def on_train_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        preds = torch.cat(self.train_preds)
        target = torch.cat(self.train_targets)

        log_metrics(pl_module=pl_module, target=target, preds=preds, prefix="train")

    def on_validation_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        log_this_epoch = self.check_if_log_every_n_epochs(trainer)

        if not log_this_epoch:
            return

        if batch_idx == 0:
            self.val_ref = []
            self.val_preds = []
            self.val_targets = []

        ref, x = batch
        predicted_samples = pl_module.sample(ref)

        self.val_ref.append(ref)
        self.val_preds.append(predicted_samples)
        self.val_targets.append(x)

    def on_validation_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        preds = torch.cat(self.val_preds)
        target = torch.cat(self.val_targets)

        log_metrics(pl_module=pl_module, target=target, preds=preds, prefix="val")

    # def on_test_batch_end(
    #    self,
    #    trainer: "L.Trainer",
    #    pl_module: "L.LightningModule",
    #    outputs: STEP_OUTPUT,
    #    batch: Any,
    #    batch_idx: int,
    #    dataloader_idx: int = 0,
    # ) -> None:

    #    if batch_idx == 0:
    #        self.test_ref = []
    #        self.test_preds = []

    #    ref, x = batch
    #    predicted_samples = pl_module.sample(ref)

    #    self.test_ref.append(ref)
    #    self.test_preds.append(predicted_samples)
    #    self.test_targets.append(x)

    def on_test_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        preds = torch.cat(pl_module.test_preds)
        target = torch.cat(pl_module.test_targets)

        log_metrics(pl_module=pl_module, target=target, preds=preds, prefix="test")


###########################
# Metrics functions below #
###########################


@torch.no_grad()
def log_metrics(*, pl_module, target, preds, prefix):
    # Calculate metrics
    psnr, std_psnr = calc_psnr(preds, target)
    ssim, std_ssim = calc_ssim(preds, target)
    mae, std_mae = calc_mae(preds, target)
    mse, std_mse = calc_mse(preds, target)

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


@torch.no_grad()
def calc_psnr(preds, targets):
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
def calc_ssim(preds, target):
    ssim = structural_similarity_index_measure(
        preds, target, reduction="none", data_range=(-1, 1)
    )
    if isinstance(ssim, tuple):
        ssim = ssim[0]

    mean_ssim = ssim.mean()
    std_ssim = ssim.std()
    return mean_ssim, std_ssim


@torch.no_grad()
def calc_mae(preds, target):
    _mae = nn.L1Loss(reduction="none")
    mae = _mae(preds, target).mean(dim=(1, 2, 3))

    mean_mae = mae.mean()
    std_mae = mae.std()

    return mean_mae, std_mae


@torch.no_grad()
def calc_mse(preds, target):
    _mse = nn.MSELoss(reduction="none")
    mse = _mse(preds, target).mean(dim=(1, 2, 3))

    mean_mse = mse.mean()
    std_mse = mse.std()

    return mean_mse, std_mse
