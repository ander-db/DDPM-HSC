from typing import Any, List, Tuple
import wandb
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib.colors import Normalize
import torch
import torch.nn as nn
import lightning as L
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio


def apply_colormap(arr, vmin, vmax, cmap="viridis"):

    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    rgba_img = colormap(norm(arr))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


def gen_grid_comparison(
    ref: torch.Tensor,
    target: torch.Tensor,
    preds: torch.Tensor,
    n_samples: int,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> Tuple[List[npt.NDArray], float, float]:
    """
    Generate a grid comparison between reference, target, and prediction tensors.

    Args:
        ref: Reference tensor
        target: Target tensor
        preds: Prediction tensor
        n_samples: Number of samples to include in grid

    Returns:
        Tuple containing:
        - List of numpy arrays forming the grid
        - Minimum value for visualization
        - Maximum value for visualization
    """
    # Convert tensors to numpy arrays
    ref_np: npt.NDArray = ref.squeeze().cpu().numpy()
    target_np: npt.NDArray = target.squeeze().cpu().numpy()
    preds_np: npt.NDArray = preds.squeeze().cpu().numpy()

    # Calculate absolute difference
    diff: npt.NDArray = np.abs(target_np - preds_np)

    # Create grid
    grid: List[npt.NDArray] = []
    for i in range(n_samples):
        grid.extend([ref_np[i], target_np[i], preds_np[i], diff[i]])

    return grid, vmin, vmax


def _log_vision_comparison(
    trainer: "L.Trainer",
    ref: torch.Tensor,
    target: torch.Tensor,
    preds: torch.Tensor,
    log_key: str,
    n_samples: int = 27,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
):
    grid, vmin, vmax = gen_grid_comparison(ref, target, preds, n_samples, vmin, vmax)

    # Apply colormap
    grid_colored = []
    for i, img in enumerate(grid):
        if (i + 1) % 4 == 0:
            img = apply_colormap(img, vmin, vmax, cmap="bwr")  # Image of the difference
        else:
            img = apply_colormap(img, vmin, vmax, cmap="cubehelix_r")
        grid_colored.append(img)

    # Generate the list of wandb images
    wandb_images = [wandb.Image(img) for img in grid_colored]

    # Log to wandb
    trainer.logger.log_image(
        log_key,
        images=wandb_images,
        step=trainer.global_step,
    )


class MetricsCallbackUNet(Callback):

    def __init__(
        self, log_visualization=False, n_visualizations=17, log_every_n_epochs=1
    ):
        super().__init__()

        self.log_visualization = log_visualization
        self.n_visualizations = n_visualizations

    def on_train_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        preds = torch.cat(pl_module.train_preds)
        target = torch.cat(pl_module.train_targets)

        log_metrics(pl_module=pl_module, target=target, preds=preds, prefix="train")

    def on_validation_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        refs = torch.cat(pl_module.val_ref)
        preds = torch.cat(pl_module.val_preds)
        target = torch.cat(pl_module.val_targets)

        log_metrics(pl_module=pl_module, target=target, preds=preds, prefix="val")

        if self.log_visualization:
            _log_vision_comparison(
                trainer=trainer,
                ref=refs,
                target=target,
                preds=preds,
                log_key="val/samples",
                n_samples=self.n_visualizations,
            )

    def on_test_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        preds = torch.cat(pl_module.test_preds)
        target = torch.cat(pl_module.test_targets)

        log_metrics(pl_module=pl_module, target=target, preds=preds, prefix="test")


class MetricsCallbackDDPM(Callback):

    def __init__(
        self, log_every_n_epochs: int = 10, log_visualization=False, n_samples=17
    ):
        super().__init__()

        self.log_every_n_epochs = log_every_n_epochs
        self.log_visualization = log_visualization
        self.n_samples = n_samples

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

        if self.log_visualization:
            _log_vision_comparison(
                trainer=trainer,
                ref=torch.cat(self.val_ref),
                target=target,
                preds=preds,
                log_key="val/samples",
                n_samples=self.n_samples,
            )

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
