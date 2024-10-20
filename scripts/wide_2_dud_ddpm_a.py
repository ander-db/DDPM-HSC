import torch
import torch.nn as nn

import lightning as L
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

from src.callbacks.metrics_new import MetricsCallbackDDPM
from src.callbacks.visual_comparison import DenoiseComparisonDDPM
from src.callbacks.metrics import LogVisionMetricsDDPM
from src.data_setup.wide_dud_v2 import Image2ImageHSCDataModule
from src.ddpm.ddpm_A import DDPM_2D
from src.utils.common_transformations import TRANSFORM_LOG_NORM_DA, TRANSFORM_LOG_NORM


if __name__ == "__main__":
    seed_everything(42)

    # Constants
    BATCH_SIZE = 32
    MAX_EPOCHS = 1_000
    LR = 2e-4  # 2e-4
    DIFF_STEPS = 50
    # ENCODER_CHANNELS = [64, 128, 256, 512]
    ENCODER_CHANNELS = [8, 16, 32, 64, 128, 256]
    # ENCODER_CHANNELS = [4, 8, 16, 32, 64, 128]
    # ENCODER_CHANNELS = [2, 4, 8, 16, 32, 64]
    DROPOUT_RATE = 0.10

    # Load, setup the Data
    data_module = Image2ImageHSCDataModule(
        mapping_dir="./data/mappings/full_seed_42_train_70_val_15/",
        batch_size=BATCH_SIZE,
        train_transform=TRANSFORM_LOG_NORM_DA,
        val_transform=TRANSFORM_LOG_NORM,
        test_transform=TRANSFORM_LOG_NORM,
        load_in_memory=True,
        num_workers=12,
    )

    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    data_module.setup("test")
    test_dataloader = data_module.test_dataloader()

    # Callbacks
    wandb_logger = WandbLogger(
        project="cvpr_image2image_test", name="unet_wide_2_dud_mae"
    )
    visual_samples = DenoiseComparisonDDPM(
        log_every_n_epochs=25, batch_idx=0, n_samples=27
    )
    metrics = LogVisionMetricsDDPM(log_every_n_epochs=15, batch_idx=-1)
    early_stopping = EarlyStopping(monitor="val/loss", patience=100)

    new_metrics = MetricsCallbackDDPM(log_every_n_epochs=50)

    # Model
    ddpm_model = DDPM_2D(
        in_channels=2,
        out_channels=1,
        diffusion_steps=DIFF_STEPS,
        dropout=DROPOUT_RATE,
        lr=LR,
        encoder_channels=ENCODER_CHANNELS,
        loss_fn=nn.MSELoss(),
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[new_metrics],
    )

    # Training

    trainer.fit(
        model=ddpm_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Testing

    trainer.test(
        model=ddpm_model,
        dataloaders=[train_dataloader],
    )

    trainer.test(
        model=ddpm_model,
        dataloaders=[val_dataloader],
    )

    trainer.test(
        model=ddpm_model,
        dataloaders=[test_dataloader],
    )
