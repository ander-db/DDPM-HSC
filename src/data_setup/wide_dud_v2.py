import os
from typing import Callable, List, Tuple
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L
from astropy.io import fits


class FITSHSCDataset(Dataset):
    def __init__(
        self,
        input_path: str,
        target_path: str,
        input_files: List[str],
        target_files: List[str],
        transform: Callable | None = None,
        load_in_memory: bool = False,
    ):
        self.input_path = input_path
        self.target_path = target_path
        self.input_files = input_files
        self.target_files = target_files
        self.transform = transform
        self.load_in_memory = load_in_memory
        self.data = self._load_data_into_memory() if self.load_in_memory else None

    def _load_fits_as_tensor(self, filepath: str) -> torch.Tensor:
        data = fits.getdata(filepath)
        if isinstance(data, np.ndarray):
            data = data.astype(np.float32)
        return torch.tensor(data, dtype=torch.float32)

    def _load_data_into_memory(self):
        return [
            (
                self._load_fits_as_tensor(
                    os.path.join(self.input_path, input_file)
                ).unsqueeze(0),
                self._load_fits_as_tensor(
                    os.path.join(self.target_path, target_file)
                ).unsqueeze(0),
            )
            for input_file, target_file in zip(self.input_files, self.target_files)
        ]

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        if self.load_in_memory:
            if self.data is None:
                raise ValueError("Data is not loaded in memory")
            input_tensor, target_tensor = self.data[idx]
        else:
            input_tensor = self._load_fits_as_tensor(
                os.path.join(self.input_path, self.input_files[idx])
            ).unsqueeze(0)
            target_tensor = self._load_fits_as_tensor(
                os.path.join(self.target_path, self.target_files[idx])
            ).unsqueeze(0)

        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        return input_tensor, target_tensor


# hsc_datamodule.py
class Image2ImageHSCDataModule(L.LightningDataModule):
    def __init__(
        self,
        input_path: str,
        target_path: str,
        mapping_file: str,
        batch_size: int = 32,
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        test_transform: Callable | None = None,
        train_size: float = 0.8,
        val_size: float = 0.1,
        load_in_memory: bool = False,
        num_workers: int = 11,
    ):
        super().__init__()
        self.input_path = input_path
        self.target_path = target_path
        self.mapping_file = mapping_file
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.train_size = train_size
        self.val_size = val_size
        self.load_in_memory = load_in_memory
        self.num_workers = num_workers

        self.input_files, self.target_files = self.read_mapping_files(self.mapping_file)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def read_mapping_files(self, mapping_file: str) -> Tuple[List[str], List[str]]:
        with open(mapping_file, "r") as f:
            lines = f.readlines()
            lines = [line.strip().split(",") for line in lines]
            return [line[0] for line in lines], [line[1] for line in lines]

    def setup(self, stage: str):
        # Shuffle and split the data
        combined = list(zip(self.input_files, self.target_files))
        random.shuffle(combined)
        self.input_files, self.target_files = zip(*combined)

        total_files = len(self.input_files)
        train_end = int(total_files * self.train_size)
        val_end = train_end + int(total_files * self.val_size)

        datasets = {
            "train": (
                self.input_files[:train_end],
                self.target_files[:train_end],
                self.train_transform,
            ),
            "val": (
                self.input_files[train_end:val_end],
                self.target_files[train_end:val_end],
                self.val_transform,
            ),
            "test": (
                self.input_files[val_end:],
                self.target_files[val_end:],
                self.test_transform,
            ),
        }

        for dataset_name, (inp_files, tgt_files, transform) in datasets.items():
            if (
                stage == "fit"
                and dataset_name in ["train", "val"]
                or stage == "test"
                and dataset_name == "test"
                or stage is None
            ):
                setattr(
                    self,
                    f"{dataset_name}_dataset",
                    FITSHSCDataset(
                        self.input_path,
                        self.target_path,
                        inp_files,
                        tgt_files,
                        transform,
                        self.load_in_memory,
                    ),
                )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Train dataset is not set up")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Val dataset is not set up ")

        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset is not set up")

        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
