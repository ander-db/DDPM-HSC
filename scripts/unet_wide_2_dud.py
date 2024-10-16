import lightning as L
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

from src.callbacks.comparison import DenoiseComparisonUNet
from src.callbacks.vision import LogVisionMetricsUNet
from src.data_setup.wide_dud_v2 import Image2ImageHSCDataModule
from src.unet.UNet_Res_GroupNorm_SiLU_D import UNet_Res_GroupNorm_SiLU_D
from src.utils.common_transformations import TRANSFORM_LOG_NORM_DA, TRANSFORM_LOG_NORM


if __name__ == "__main__":
    seed_everything(42)

    # Constants
    BATCH_SIZE = 16
    MAX_EPOCHS = 5
    LR = 2e-4
    ENCODER_CHANNELS = [64, 128, 256, 512]
    DROPOUT_RATE = 0.10

    # Load, setup the Data
    data_module = Image2ImageHSCDataModule(
        mapping_dir="./data/mappings/testing_mapping/",
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
        project="cvpr_image2image_test", name="unet_wide_2_dud_init"
    )
    visual_samples = DenoiseComparisonUNet(
        log_every_n_epochs=5, batch_idx=0, n_samples=27
    )
    metrics = LogVisionMetricsUNet(log_every_n_epochs=5, batch_idx=-1)
    early_stopping = EarlyStopping(monitor="val/loss", patience=100)

    # Model
    unet_model = UNet_Res_GroupNorm_SiLU_D(
        in_channels=1,
        out_channels=1,
        lr=LR,
        encoder_channels=ENCODER_CHANNELS,
        dropout_rate=DROPOUT_RATE,
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[visual_samples, metrics, early_stopping],
    )

    trainer.fit(unet_model, train_dataloader, val_dataloader)

    trainer.test(unet_model, test_dataloader)
