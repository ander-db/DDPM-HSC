from lightning import seed_everything
import matplotlib.pyplot as plt
from src.data_setup.wide_dud_v2 import Image2ImageHSCDataModule
from src.utils.common_transformations import TRANSFORM_LOG_NORM_DA


def normalize_to_range(tensor, min_val=-1, max_val=1):
    min_tensor = tensor.min()
    max_tensor = tensor.max()
    return (tensor - min_tensor) / (max_tensor - min_tensor) * (
        max_val - min_val
    ) + min_val


def plot_normalized_images(input_tensor, target_tensor, num_samples=32):
    # Calculate the number of rows and columns for the grid
    num_cols = min(8, num_samples)  # Maximum 8 columns
    num_rows = (num_samples + num_cols - 1) // num_cols  # Ceiling division

    # Create a figure with no internal padding
    fig = plt.figure(figsize=(2.5 * num_cols, 5 * num_rows))

    for i in range(num_samples):
        for j, (img, title) in enumerate(
            zip([input_tensor[i, 0], target_tensor[i, 0]], ["Input", "Target"])
        ):
            # Calculate the position of each image in the grid
            row = (2 * i + j) // num_cols
            col = (2 * i + j) % num_cols

            # Add a subplot with no spacing
            ax = fig.add_subplot(2 * num_rows, num_cols, 2 * i + j + 1)

            # Display the image
            ax.imshow(img, cmap="cubehelix_r", vmin=-1, vmax=1)
            # im = axs[j, i].imshow(img, cmap="inferno", vmin=-1, vmax=1)

            # Remove axes and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

    # Remove any spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    # Ensure the plot fills the entire figure
    plt.tight_layout(pad=0)

    plt.show()


if __name__ == "__main__":
    seed_everything(42)

    # Modify your transformation to include normalization to [-1, 1] range
    transformation = TRANSFORM_LOG_NORM_DA

    N_SAMPLES = 32

    data_module = Image2ImageHSCDataModule(
        mapping_dir="./data/mappings/testing_mapping/",
        batch_size=N_SAMPLES,
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
    plot_normalized_images(input, target, num_samples=N_SAMPLES)
