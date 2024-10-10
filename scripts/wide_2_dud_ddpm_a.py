import lightning as L
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger

from src.ddpm.ddpm_A import DDPM_2D
from src.data_setup.wide_dud_v2 import Image2ImageHSCDataModule
from src.utils.common_transformations import TRANSFORM_LOG_NORM_DA, TRANSFORM_LOG_NORM


if __name__ == "__main__":
    # Seed everything
    seed_everything(42)

    # Constants
    BATCH_SIZE = 16
    DIFF_STEPS = 5
    MAX_EPOCHS = 250
    LR = 5e-4
    ENCODER_CHANNELS = [16, 32, 64, 128]

    # Load, setup the Data
    data_module = Image2ImageHSCDataModule(
        mapping_dir="./data/mappings/testing_mapping/",
        batch_size=BATCH_SIZE,
        train_transform=TRANSFORM_LOG_NORM_DA,
        val_transform=TRANSFORM_LOG_NORM,
        test_transform=TRANSFORM_LOG_NORM,
        load_in_memory=False,
        num_workers=11,
    )
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    sample_train = next(iter(train_dataloader))
    sample_val = next(iter(val_dataloader))

    print("Sample Train Shape:", sample_train[0].shape, sample_train[1].shape)
    print("Sample Val Shape:", sample_val[0].shape, sample_val[1].shape)

    # Model
    ddpm_model = DDPM_2D(
        in_channels=2,
        out_channels=1,
        diffusion_steps=DIFF_STEPS,
        lr=LR,
        encoder_channels=ENCODER_CHANNELS,
    )

    # Callback
    wandb_logger = WandbLogger(project="cvpr_image2image_test")

    # Trainer
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS, logger=wandb_logger, log_every_n_steps=10
    )

    # Trainer Fit
    trainer.fit(ddpm_model, train_dataloader, val_dataloader)

    # Call sample plot
    data_module.setup("test")
    test_dataloader = data_module.test_dataloader()

    # Predict
    ref_test, expected_test = next(iter(test_dataloader))
    ref_test = ref_test.to("cuda")
    expected_test = expected_test.to("cuda")
    ddpm_model = ddpm_model.to("cuda")

    print(f'ref_test device: {ref_test.device}')
    print(f'expected_test device: {expected_test.device}')

    pred = ddpm_model.sample(ref_test)

    # Plot ref, expected, pred
    def plot_samples(ref, expected, pred):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        for i, (img, title) in enumerate(
            zip([ref[0, 0], expected[0, 0], pred[0, 0]], ["Ref", "Expected", "Pred"])
        ):
            axs[i].imshow(img, cmap="cubehelix_r", vmin=-1, vmax=1)
            axs[i].set_title(title)
            axs[i].axis("off")

        plt.show()

    plot_samples(ref_test.cpu(), expected_test.cpu(), pred.cpu())
