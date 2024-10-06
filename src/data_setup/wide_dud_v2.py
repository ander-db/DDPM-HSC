import os
from typing import Callable, List, Tuple
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L
from astropy.io import fits


class FITSHSCDataset(Dataset):
    def __init__(
        self,
        input_files: List[str],
        target_files: List[str],
        transform: Callable | None = None,
        load_in_memory: bool = False,
    ):
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
                self._load_fits_as_tensor(input_file).unsqueeze(0),
                self._load_fits_as_tensor(target_file).unsqueeze(0),
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
            input_tensor = self._load_fits_as_tensor(self.input_files[idx]).unsqueeze(0)
            target_tensor = self._load_fits_as_tensor(self.target_files[idx]).unsqueeze(
                0
            )

        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        return input_tensor, target_tensor


class Image2ImageHSCDataModule(L.LightningDataModule):
    def __init__(
        self,
        mapping_dir: str,
        batch_size: int = 32,
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        test_transform: Callable | None = None,
        load_in_memory: bool = False,
        num_workers: int = 11,
    ):
        super().__init__()
        self.mapping_dir = mapping_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.load_in_memory = load_in_memory
        self.num_workers = num_workers

        self.train_files = self.read_mapping_file(
            os.path.join(mapping_dir, "train_files.txt")
        )
        self.val_files = self.read_mapping_file(
            os.path.join(mapping_dir, "val_files.txt")
        )
        self.test_files = self.read_mapping_file(
            os.path.join(mapping_dir, "test_files.txt")
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def read_mapping_file(self, mapping_file: str) -> List[Tuple[str, str]]:
        with open(mapping_file, "r") as f:
            lines = f.readlines()
            return [(line.split(",")[0], line.split(",")[1][:-1]) for line in lines]

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train_dataset = FITSHSCDataset(
                input_files=[f[0] for f in self.train_files],
                target_files=[f[1] for f in self.train_files],
                transform=self.train_transform,
                load_in_memory=self.load_in_memory,
            )
            self.val_dataset = FITSHSCDataset(
                input_files=[f[0] for f in self.val_files],
                target_files=[f[1] for f in self.val_files],
                transform=self.val_transform,
                load_in_memory=self.load_in_memory,
            )

        if stage == "test" or stage is None:
            self.test_dataset = FITSHSCDataset(
                input_files=[f[0] for f in self.test_files],
                target_files=[f[1] for f in self.test_files],
                transform=self.test_transform,
                load_in_memory=self.load_in_memory,
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
            raise ValueError("Val dataset is not set up")
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset is not set up")
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
