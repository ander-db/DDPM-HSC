from lightning import seed_everything
import torch
import matplotlib.pyplot as plt
from src.data_setup.wide_dud_v2 import Image2ImageHSCDataModule
from src.utils.common_transformations import TRANSFORM_LOG_NORM_FLIP
import numpy as np


def normalize_to_range(tensor, min_val=-1, max_val=1):
    min_tensor = tensor.min()
    max_tensor = tensor.max()
    return (tensor - min_tensor) / (max_tensor - min_tensor) * (
        max_val - min_val
    ) + min_val


def plot_normalized_images(input_tensor, target_tensor, num_samples=8):
    fig, axs = plt.subplots(2, num_samples, figsize=(2 * num_samples, 10))
    im = None

    for i in range(num_samples):
        for j, (img, title) in enumerate(
            zip([input_tensor[i, 0], target_tensor[i, 0]], ["Input", "Target"])
        ):
            #im = axs[j, i].imshow(img, cmap="cubehelix_r", vmin=-1, vmax=1)
            im = axs[j, i].imshow(img, cmap="inferno", vmin=-1, vmax=1)
            axs[j, i].axis("off")
            axs[j, i].set_title(title)

    if im is None:
        raise ValueError("No images to plot")

    # Add a colorbar
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    seed_everything(42)

    # Modify your transformation to include normalization to [-1, 1] range
    transformation = (
        TRANSFORM_LOG_NORM_FLIP  # Ensure this includes normalization to [-1, 1]
    )

    data_module = Image2ImageHSCDataModule(
        mapping_dir="./data/mappings/testing_mapping/",
        batch_size=8,
        train_transform=transformation,
        val_transform=transformation,
        test_transform=transformation,
        load_in_memory=False,
        num_workers=0,
    )
    data_module.setup("fit")
    dataloader = data_module.val_dataloader()
    input, target = next(iter(dataloader))

    # Ensure data is in [-1, 1] range
    input = normalize_to_range(input)
    target = normalize_to_range(target)

    print("Input shape:", input.shape)
    print("Target shape:", target.shape)

    # Plot the normalized data
    plot_normalized_images(input, target)
