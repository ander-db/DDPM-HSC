import torch
import lightning as L
from lightning.fabric.utilities.seed import seed_everything

from src.data_setup.wide_dud_v2 import Image2ImageHSCDataModule
from src.utils.common_transformations import TRANSFORM_LOG_NORM_DA, TRANSFORM_LOG_NORM

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

if __name__ == "__main__":
    # Seed everything
    seed_everything(42)

    # Constants
    BATCH_SIZE = 16

    DIFFUSION_STEPS = 10
    BETA_START = 0.000
    BETA_END = 0.2
    SCHEDULER = "squaredcos_cap_v2"

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

    sample = next(iter(train_dataloader))
    ref, x = sample
    print(f"Max value of x: {x.max()}")
    print(f"Min value of x: {x.min()}")

    noise_scheduler = DDPMScheduler(DIFFUSION_STEPS, BETA_START, BETA_END, SCHEDULER)
    noise_scheduler.set_timesteps(DIFFUSION_STEPS)

    noisy_images = []

    noisy_images.append(x.clone())

    # Apply noise scheduler and plot a grid with different noise levels
    for t in range(DIFFUSION_STEPS):
        noise = torch.rand_like(x)

        timesteps = torch.tensor([t], dtype=torch.int).repeat(x.shape[0])
        noisy_image = noise_scheduler.add_noise(x, noise, timesteps)
        noisy_images.append(noisy_image)

    # Function to display images in a row
    def show_images_row(images, titles):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img.squeeze()[1].cpu().numpy(), cmap="inferno")
            ax.set_title(title)
            ax.axis("off")

        plt.show()

    # Display the noisy images
    show_images_row(
        noisy_images,
        ["X original"] + [f"Noise level: {t}" for t in range(0, DIFFUSION_STEPS)],
    )

    # Plot the difference between the original and the noisy_images[1]
    diff = x - noisy_images[1]
    import matplotlib.pyplot as plt

    plt.imshow(diff.squeeze()[1].cpu().numpy(), cmap="inferno", vmin=-0.2, vmax=0.2)
    plt.colorbar()
    plt.axis("off")
    plt.show()
