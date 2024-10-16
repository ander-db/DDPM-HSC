import torch
import wandb
import logging
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import Callback

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def apply_colormap(arr, vmin, vmax, cmap="viridis"):
    """
    Aplica un colormap a una imagen en escala de grises

    Args:
    - arr (np.ndarray): Imagen en escala de grises
    - vmin (float): Valor mínimo de la Imagen
    - vmax (float): Valor máximo de la Imagen
    - cmap (str): Nombre del colormap a aplicar

    Returns:
    - np.ndarray: Imagen en RGB con el colormap aplicado
    """

    # Aplica el colormap 'rainbow' y convierte el resultado a RGB
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    rgba_img = colormap(norm(arr))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


class DenoiseComparisonBase(Callback):
    def __init__(
        self,
        log_every_n_epochs: int = 20,
        n_samples: int = 27,
        batch_idx: int = 0,
        transform: str | None = None,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.n_samples = n_samples
        self.batch_idx = batch_idx
        self.transform = transform

    @staticmethod
    def gen_grid_comparison(
        ref: torch.Tensor, target: torch.Tensor, preds: torch.Tensor, n_samples: int
    ):
        ref = ref.squeeze().cpu().numpy()
        target = target.squeeze().cpu().numpy()
        preds = preds.squeeze().cpu().numpy()
        diff = np.abs(target - preds)

        vmin, vmax = -1, 1

        grid = []
        for i in range(n_samples):
            grid.extend([ref[i], target[i], preds[i], diff[i]])

        return grid, vmin, vmax

    @torch.no_grad()
    def calc_preds(self, pl_module: "L.LightningModule", ref: torch.Tensor):
        raise NotImplementedError("Subclasses must implement this method")

    def on_validation_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        self._log_comparison(trainer, pl_module, trainer.val_dataloaders, "val/samples")

    # def on_train_end(
    #    self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    # ) -> None:
    #    self._log_comparison(
    #        trainer,
    #        pl_module,
    #        trainer.train_dataloader,
    #        "sample_wide_dud_pred_error_train",
    #    )

    def _log_comparison(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        dataloader,
        log_key: str,
    ) -> None:
        if not self._should_log(trainer):
            return

        batch = self._get_batch(dataloader)
        if batch is None:
            return

        list_of_images = self._process_batch(pl_module, batch)
        if list_of_images:
            self._log_to_wandb(trainer, list_of_images, log_key)

    def _should_log(self, trainer: "L.Trainer") -> bool:
        return (
            trainer.current_epoch % self.log_every_n_epochs == 0
            and trainer.current_epoch != 0
        )

    def _get_batch(self, dataloader):
        if dataloader is None:
            return None
        for i, batch in enumerate(dataloader):
            if i == self.batch_idx:
                return batch
        return None

    def _process_batch(self, pl_module: "L.LightningModule", batch):
        ref, target = batch
        ref = ref.to(pl_module.device)
        target = target.to(pl_module.device)
        preds = self.calc_preds(pl_module=pl_module, ref=ref)

        if preds is None:
            return []

        self._adjust_n_samples(ref)
        grid, vmin, vmax = self.gen_grid_comparison(ref, target, preds, self.n_samples)
        return self._create_image_list(grid)

    def _adjust_n_samples(self, ref: torch.Tensor) -> None:
        if self.n_samples > ref.shape[0]:
            self.n_samples = ref.shape[0]
            logging.warning(
                f"The number of samples is greater than the batch size. Setting n_samples to {self.n_samples}"
            )

    def _create_image_list(self, grid):
        list_of_images = []
        for i, image in enumerate(grid):
            if (i + 1) % 4 == 0:
                list_of_images.append(
                    wandb.Image(apply_colormap(image, -1, +1, cmap="bwr"))
                )
            else:
                list_of_images.append(wandb.Image(apply_colormap(image, -1, +1)))
        return list_of_images

    def _log_to_wandb(self, trainer: "L.Trainer", images, log_key: str) -> None:
        if trainer.logger is not None:
            trainer.logger.log_image(
                log_key,
                images=images,
                step=trainer.global_step,
            )


class DenoiseComparisonDDPM(DenoiseComparisonBase):
    def __init__(
        self,
        log_every_n_epochs: int = 20,
        n_samples: int = 27,
        batch_idx: int = 0,
        transform: str | None = None,
    ):
        super().__init__(log_every_n_epochs, n_samples, batch_idx, transform)

    @torch.no_grad()
    def calc_preds(self, pl_module: "L.LightningModule", ref: torch.Tensor):
        return pl_module.sample(ref.to(pl_module.device))


class DenoiseComparisonUNet(DenoiseComparisonBase):
    def __init__(
        self,
        log_every_n_epochs: int = 20,
        n_samples: int = 27,
        batch_idx: int = 0,
        transform: str | None = None,
    ):
        super().__init__(log_every_n_epochs, n_samples, batch_idx, transform)

    @torch.no_grad()
    def calc_preds(self, pl_module: "L.LightningModule", ref: torch.Tensor):
        return pl_module.forward(ref.to(pl_module.device), t=None)
