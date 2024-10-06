import os
import random
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.model_selection import KFold
from astropy.io import fits
import torch
from torchmetrics.image import PeakSignalNoiseRatio


def write_mapping_file(filename: str, data: List[Tuple[str, str, float]]):
    """Write mapping data to a file."""
    with open(filename, "w") as f:
        for item in data:
            f.write(f"{item[0]},{item[1]}\n")


def setup_logging(log_file: str):
    """Set up logging configuration to overwrite the log file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        filemode="w",  # 'w' mode overwrites the file
    )


def log_and_print(message: str):
    """Log a message and print it to console."""
    logging.info(message)
    print(message)


def load_fits_as_tensor(filepath: str) -> torch.Tensor:
    """
    Load a FITS file as a PyTorch tensor.

    Args:
        filepath (str): Path to the FITS file.

    Returns:
        torch.Tensor: The FITS file as a PyTorch tensor.
    """
    data = fits.getdata(filepath)
    if isinstance(data, np.ndarray):
        data = data.astype(np.float32)

    return torch.tensor(data, dtype=torch.float32)


def calculate_psnr(
    input_dir: str, target_dir: str, input_files: List[str], target_files: List[str]
) -> List[Tuple[str, str, float]]:
    """Calculate PSNR for each pair of input and target files."""
    psnr_metric = PeakSignalNoiseRatio()
    psnr_values = []
    for input_file, target_file in zip(input_files, target_files):
        input_tensor = load_fits_as_tensor(
            os.path.join(input_dir, input_file)
        ).unsqueeze(0)
        target_tensor = load_fits_as_tensor(
            os.path.join(target_dir, target_file)
        ).unsqueeze(0)
        psnr = psnr_metric(target_tensor, input_tensor).item()
        psnr_values.append((input_file, target_file, psnr))
    return psnr_values


def generate_psnr_histogram(
    all_psnr_values: List[float],
    split_psnr_values: List[List[float]],
    split_names: List[str],
    output_file: str,
):
    """Generate and save a stacked histogram of PSNR values with cubehelix color palette."""
    plt.figure(figsize=(9, 5))

    # Set the style
    plt.style.use("seaborn-v0_8-darkgrid")

    # Use cubehelix color palette
    colors = [
        "#42A5F5",  # Medium blue
        "#1E88E5",  # Dark blue
        "#0D47A1",  # Very dark blue
    ]

    # Define the bins and convert to list
    bins = list(np.linspace(min(all_psnr_values), max(all_psnr_values), 50))

    # Plot stacked histogram for splits
    n, bins, patches = plt.hist(
        split_psnr_values, bins=bins, stacked=True, label=split_names, color=colors
    )

    # Transform the bins into a  Sequence[float]
    bins = list(bins)

    plt.xlabel("PSNR (dB)", fontsize=12, fontweight="bold")
    plt.ylabel("Frequency", fontsize=12, fontweight="bold")
    plt.title("PSNR Distribution Across Dataset Splits", fontsize=16, fontweight="bold")

    # Customize the legend
    legend = plt.legend(loc="upper left", frameon=True, fontsize=12)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("gray")

    # Customize the axes
    plt.tick_params(axis="both", which="major", labelsize=10)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def generate_mapping_files(
    input_dir: str,
    target_dir: str,
    output_dir: str,
    split_name: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    cross_validation: int | None = None,
    psnr_threshold: float | None = None,
):
    """
    Generate mapping files for train, validation, and test sets, with optional cross-validation.
    Also generates a PSNR distribution histogram.
    """
    # Set up logging at the beginning
    if cross_validation is None:
        split_dir = os.path.join(output_dir, split_name)
    else:
        split_dir = (
            output_dir  # For cross-validation, we'll create fold-specific logs later
        )
    os.makedirs(split_dir, exist_ok=True)
    log_file = os.path.join(split_dir, "logs.txt")
    setup_logging(log_file)

    input_files = [f for f in os.listdir(input_dir) if f.endswith(".fits")]
    target_files = [f for f in os.listdir(target_dir) if f.endswith(".fits")]

    assert len(input_files) == len(
        target_files
    ), "Input and target files must have the same length"

    # Calculate PSNR for all files
    all_psnr_data = calculate_psnr(input_dir, target_dir, input_files, target_files)

    # Filter based on PSNR threshold if specified
    if psnr_threshold is not None:
        original_count = len(all_psnr_data)
        all_psnr_data = [item for item in all_psnr_data if item[2] >= psnr_threshold]
        filtered_count = len(all_psnr_data)
        log_and_print(f"Filtered data based on PSNR threshold: {psnr_threshold}")
        log_and_print(
            f"Original count: {original_count}, After filtering: {filtered_count}"
        )

    all_psnr_values = [item[2] for item in all_psnr_data]

    # Set random seed
    random.seed(seed)

    # Shuffle the data
    random.shuffle(all_psnr_data)

    total = len(all_psnr_data)

    common_log_messages = [
        f"Input directory: {input_dir}",
        f"Target directory: {target_dir}",
        f"Seed: {seed}",
        f"Total files after filtering: {total}",
    ]

    if cross_validation is None:
        # Regular split
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)

        train_data = all_psnr_data[:train_end]
        val_data = all_psnr_data[train_end:val_end]
        test_data = all_psnr_data[val_end:]

        write_mapping_file(os.path.join(split_dir, "train_files.txt"), train_data)
        write_mapping_file(os.path.join(split_dir, "val_files.txt"), val_data)
        write_mapping_file(os.path.join(split_dir, "test_files.txt"), test_data)

        # Generate PSNR histogram
        train_psnr = [item[2] for item in train_data]
        val_psnr = [item[2] for item in val_data]
        test_psnr = [item[2] for item in test_data]
        generate_psnr_histogram(
            all_psnr_values,
            [train_psnr, val_psnr, test_psnr],
            ["Train", "Validation", "Test"],
            os.path.join(split_dir, "psnr_distribution.svg"),
        )

        log_messages = common_log_messages + [
            f"Split type: Regular (Train/Validation/Test)",
            f"Train ratio: {train_ratio:.2f}",
            f"Validation ratio: {val_ratio:.2f}",
            f"Test ratio: {1 - train_ratio - val_ratio:.2f}",
            f"Train files: {len(train_data)}",
            f"Validation files: {len(val_data)}",
            f"Test files: {len(test_data)}",
        ]

        for message in log_messages:
            log_and_print(message)

    else:
        # Cross-validation split
        kf = KFold(n_splits=cross_validation, shuffle=True, random_state=seed)

        for fold, (train_val_index, test_index) in enumerate(
            kf.split(all_psnr_data), 1
        ):
            fold_dir = os.path.join(output_dir, f"{split_name}_cv_{fold}")
            os.makedirs(fold_dir, exist_ok=True)
            log_file = os.path.join(fold_dir, "logs.txt")
            setup_logging(log_file)

            train_val_data = [all_psnr_data[i] for i in train_val_index]
            test_data = [all_psnr_data[i] for i in test_index]

            # Further split train_val into train and validation
            train_end = int(
                len(train_val_data) * (train_ratio / (train_ratio + val_ratio))
            )
            train_data = train_val_data[:train_end]
            val_data = train_val_data[train_end:]

            write_mapping_file(os.path.join(fold_dir, "train_files.txt"), train_data)
            write_mapping_file(os.path.join(fold_dir, "val_files.txt"), val_data)
            write_mapping_file(os.path.join(fold_dir, "test_files.txt"), test_data)

            # Generate PSNR histogram for this fold
            train_psnr = [item[2] for item in train_data]
            val_psnr = [item[2] for item in val_data]
            test_psnr = [item[2] for item in test_data]
            generate_psnr_histogram(
                all_psnr_values,
                [train_psnr, val_psnr, test_psnr],
                ["Train", "Validation", "Test"],
                os.path.join(fold_dir, "psnr_distribution.png"),
            )

            log_messages = common_log_messages + [
                f"Split type: {cross_validation}-Fold Cross-Validation",
                f"Current fold: {fold}/{cross_validation}",
                f"Train ratio within fold: {train_ratio / (train_ratio + val_ratio):.2f}",
                f"Validation ratio within fold: {val_ratio / (train_ratio + val_ratio):.2f}",
                f"Train files: {len(train_data)}",
                f"Validation files: {len(val_data)}",
                f"Test files: {len(test_data)}",
            ]

            for message in log_messages:
                log_and_print(message)

    print(f"Mapping files and PSNR histograms generated in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mapping files for dataset splits"
    )
    parser.add_argument(
        "--input_dir",
        default="./data/wide_64x64",
        help="Directory containing input FITS files. Default: ./data/wide_64x64",
    )
    parser.add_argument(
        "--target_dir",
        default="./data/dud_64x64",
        help="Directory containing target FITS files. Default: ./data/dud_64x64",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save the mapping files"
    )
    parser.add_argument(
        "--split_name", required=True, help="Name of the split (e.g., 'split_3')"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data. Default: 0.8",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of validation data. Default: 0.1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--cross_validation",
        type=int,
        default=None,
        help="Number of folds for cross-validation. Default: None (no cross-validation)",
    )
    parser.add_argument(
        "--psnr_threshold",
        type=float,
        default=None,
        help="Minimum PSNR value to include in the dataset. Default: None (include all)",
    )

    args = parser.parse_args()

    generate_mapping_files(
        args.input_dir,
        args.target_dir,
        args.output_dir,
        args.split_name,
        args.train_ratio,
        args.val_ratio,
        args.seed,
        args.cross_validation,
        args.psnr_threshold,
    )
