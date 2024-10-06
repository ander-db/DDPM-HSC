import os
from typing import Callable
import torch
import random
import numpy as np
import lightning as L
import torch.utils.data
from torch.utils.data import Dataset, DataLoader


class ImageToImageHSCDataset(Dataset):
    """
    Dataset for the HSC dataset.
    """

    def __init__(
        self,
        input_path,
        target_path,
        input_files,
        target_files,
        transform=None,
        load_in_memory=False,
    ):
        """
        Args:
            input_path (str): Path to the input data.
            target_path (str): Path to the target data.
            input_files (list): List of input files.
            target_files (list): List of target files.
            transform (callable, optional): Transformation to apply to the data. Default: None.
            load_in_memory (bool, optional): Whether to load the data into memory. Default: False.
        """
        self.input_path = input_path
        self.target_path = target_path
        self.input_files = input_files
        self.target_files = target_files
        self.transform = transform
        self.load_in_memory = load_in_memory

        if self.load_in_memory:
            self.data = self._load_data_into_memory()

    def _load_data_into_memory(self):
        data = []
        for input_file, target_file in zip(self.input_files, self.target_files):
            input_tensor = torch.load(os.path.join(self.input_path, input_file))
            target_tensor = torch.load(os.path.join(self.target_path, target_file))
            data.append((input_tensor, target_tensor))
        return data

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        if self.load_in_memory:
            input_tensor, target_tensor = self.data[idx]
            input_tensor = input_tensor.unsqueeze(0)
            target_tensor = target_tensor.unsqueeze(0)
        else:
            input_file = os.path.join(self.input_path, self.input_files[idx])
            target_file = os.path.join(self.target_path, self.target_files[idx])
            input_tensor = torch.load(input_file).unsqueeze(0)
            target_tensor = torch.load(target_file).unsqueeze(0)

        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        return input_tensor.float(), target_tensor.float()


class ImageToImageHSCDatasetFITS(Dataset):
    """
    The same as ImageToImageHSCDatasetFITS but it receives a FITS directory instead of a tensor directory
    """

    def __init__(
        self,
        input_path,
        target_path,
        input_files,
        target_files,
        transform=None,
        load_in_memory=False,
    ):
        """
        Args:
            input_path (str): Path to the input data.
            target_path (str): Path to the target data.
            input_files (list): List of input files.
            target_files (list): List of target files.
            transform (callable, optional): Transformation to apply to the data. Default: None.
            load_in_memory (bool, optional): Whether to load the data into memory. Default: False.
        """
        self.input_path = input_path
        self.target_path = target_path
        self.input_files = input_files
        self.target_files = target_files
        self.transform = transform
        self.load_in_memory = load_in_memory

        if self.load_in_memory:
            self.data = self._load_data_into_memory()

    def _load_fits_as_tensor(self, filepath, channel=0):
        """
        Load a FITS file as a PyTorch tensor.

        Args:
            filepath (str): Path to the FITS file.
            channel (int): Channel to load. Default: 0.

        Returns:
            torch.Tensor: The FITS file as a PyTorch tensor.
        """
        from astropy.io import fits

        data = fits.getdata(filepath)
        if isinstance(data, np.ndarray):
            data = data.astype(np.float32)

        return torch.tensor(data, dtype=torch.float32)

    def _load_data_into_memory(self):
        """
        Load the data into memory
        """
        data = []
        for input_file, target_file in zip(self.input_files, self.target_files):
            input_tensor = self._load_fits_as_tensor(
                os.path.join(self.input_path, input_file)
            ).unsqueeze(0)
            target_tensor = self._load_fits_as_tensor(
                os.path.join(self.target_path, target_file)
            ).unsqueeze(0)
            data.append((input_tensor, target_tensor))
        return data

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        if self.load_in_memory:
            input_tensor, target_tensor = self.data[idx]
        else:
            input_file = os.path.join(self.input_path, self.input_files[idx])
            target_file = os.path.join(self.target_path, self.target_files[idx])
            input_tensor = self._load_fits_as_tensor(input_file).unsqueeze(0)
            target_tensor = self._load_fits_as_tensor(target_file).unsqueeze(0)

        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        return input_tensor, target_tensor


class ImageToImageHSCDataModule(L.LightningDataModule):
    """
    Data module for the HSC dataset. The module splits the data into training, testing, and validation sets.
    """

    def __init__(
        self,
        input_path,
        target_path,
        batch_size=32,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        train_size=0.8,
        val_size=0.1,
        load_in_memory=False,
    ):
        """
        Args:
            input_path (str): Path to the input data.
            target_path (str): Path to the target data.
            batch_size (int): Batch size. Default: 32.
            train_transform (callable, optional): Transformation to apply to the training data. Default: None.
            val_transform (callable, optional): Transformation to apply to the validation data. Default: None.
            test_transform (callable, optional): Transformation to apply to the test data. Default: None.
            train_size (float): Size of the training set. Default: 0.8.
            val_size (float): Size of the validation set. Default: 0.1.
            load_in_memory (bool, optional): Whether to load the data into memory. Default: False.
        """
        super().__init__()
        self.input_path = input_path
        self.target_path = target_path
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.train_size = train_size
        self.val_size = val_size
        self.load_in_memory = load_in_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        """
        Setup the data module.

        Args:
            stage (str): Stage of the data module. Options: "fit", "test", "val".
        """
        input_files = sorted(os.listdir(self.input_path))
        target_files = sorted(os.listdir(self.target_path))

        assert len(input_files) == len(
            target_files
        ), "Input and target files do not match"

        combined = list(zip(input_files, target_files))
        random.shuffle(combined)
        input_files, target_files = zip(*combined)

        total_files = len(input_files)
        train_end = int(total_files * self.train_size)
        val_end = train_end + int(total_files * self.val_size)

        train_input_files = list(input_files[:train_end])
        val_input_files = list(input_files[train_end:val_end])
        test_input_files = list(input_files[val_end:])

        train_target_files = list(target_files[:train_end])
        val_target_files = list(target_files[train_end:val_end])
        test_target_files = list(target_files[val_end:])

        if stage == "fit" or stage is None:
            self.train_dataset = ImageToImageHSCDataset(
                self.input_path,
                self.target_path,
                train_input_files,
                train_target_files,
                self.train_transform,
                self.load_in_memory,
            )

            self.val_dataset = ImageToImageHSCDataset(
                self.input_path,
                self.target_path,
                val_input_files,
                val_target_files,
                self.val_transform,
                self.load_in_memory,
            )

        if stage == "test" or stage is None:
            self.test_dataset = ImageToImageHSCDataset(
                self.input_path,
                self.target_path,
                test_input_files,
                test_target_files,
                self.test_transform,
                self.load_in_memory,
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Train dataset is not set.")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not set.")
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset is not set.")
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class ImageToImageHSCBaseDataModule(L.LightningDataModule):
    """
    Data module for the HSC dataset. The module splits the data into training, testing, and validation sets.
    """

    def __init__(
        self,
        input_path,
        target_path,
        input_files,
        target_files,
        batch_size=32,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        train_size=0.8,
        val_size=0.1,
        load_in_memory=False,
    ):
        super().__init__()
        self.input_path = input_path
        self.target_path = target_path
        self.input_files = input_files
        self.target_files = target_files

        assert len(self.input_files) == len(
            self.target_files
        ), "Input and target files do not match"

        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.train_size = train_size
        self.val_size = val_size
        self.load_in_memory = load_in_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _split_train_test_val(self):
        assert len(self.input_files) == len(
            self.target_files
        ), "Input and target files do not match"

        # Random shuffle
        combined = list(zip(self.input_files, self.target_files))

        random.shuffle(combined)
        input_files, target_files = zip(*combined)

        # splits
        total_files = len(input_files)
        train_end = int(total_files * self.train_size)
        val_end = train_end + int(total_files * self.val_size)

        train_input_files = list(input_files[:train_end])
        val_input_files = list(input_files[train_end:val_end])
        test_input_files = list(input_files[val_end:])
        train_target_files = list(target_files[:train_end])
        val_target_files = list(target_files[train_end:val_end])
        test_target_files = list(target_files[val_end:])
        return (
            train_input_files,
            val_input_files,
            test_input_files,
            train_target_files,
            val_target_files,
            test_target_files,
        )

    def setup(self, stage: str):
        (
            train_input_files,
            val_input_files,
            test_input_files,
            train_target_files,
            val_target_files,
            test_target_files,
        ) = self._split_train_test_val()

        if stage == "fit" or stage is None:
            self.train_dataset = ImageToImageHSCDatasetFITS(
                self.input_path,
                self.target_path,
                train_input_files,
                train_target_files,
                self.train_transform,
                self.load_in_memory,
            )

            self.val_dataset = ImageToImageHSCDatasetFITS(
                self.input_path,
                self.target_path,
                val_input_files,
                val_target_files,
                self.val_transform,
                self.load_in_memory,
            )

        if stage == "test" or stage is None:
            self.test_dataset = ImageToImageHSCDatasetFITS(
                self.input_path,
                self.target_path,
                test_input_files,
                test_target_files,
                self.test_transform,
                self.load_in_memory,
            )

    def read_mapping_files(self, mapping_file):
        """
        Read the mapping file and return the input and target files

        Args:
            mapping_file (str): Path to the mapping file

        Returns:
            input_files (list): List of input files
        """
        with open(mapping_file, "r") as f:
            lines = f.readlines()
            lines = [line.strip().split(",") for line in lines]

            input_files = [line[0] for line in lines]
            target_files = [line[1] for line in lines]

        return input_files, target_files

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Train dataset is not set.")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=11)

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not set.")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=11)

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset is not set.")
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class ImageToImageHSCFilenamesDataModule(ImageToImageHSCBaseDataModule):
    """
    Data module for the HSC dataset. The module splits the data into training, testing, and validation sets.

    Args:
        input_path (str): Path to the input data.
        target_path (str): Path to the target data.
        input_files (list): List of input files.
        target_files (list): List of target files.
        batch_size (int): Batch size. Default: 32.
        train_transform (callable, optional): Transformation to apply to the training data. Default: None.
        val_transform (callable, optional): Transformation to apply to the validation data. Default: None.
        test_transform (callable, optional): Transformation to apply to the test data. Default: None.
        train_size (float): Size of the training set. Default: 0.8.
        val_size (float): Size of the validation set. Default: 0.1.
        load_in_memory (bool, optional): Whether to load the data into memory. Default: False.
    """

    def __init__(
        self,
        input_path,
        target_path,
        input_files,
        target_files,
        batch_size=32,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        train_size=0.8,
        val_size=0.1,
        load_in_memory=False,
    ):
        super().__init__(
            input_path,
            target_path,
            input_files,
            target_files,
            batch_size,
            train_transform,
            val_transform,
            test_transform,
            train_size,
            val_size,
            load_in_memory,
        )


class Image2ImageHSCDataModule(ImageToImageHSCBaseDataModule):
    """
    Data module for the HSC dataset. The module splits the data into training, testing, and validation sets.

    Args:
        input_path (str): Path to the input data.
        target_path (str): Path to the target data.
        mapping_file (str): Path to the mapping file. The mapping file should contain the input and target filenames.
        batch_size (int): Batch size. Default: 32.
        train_transform (callable, optional): Transformation to apply to the training data. Default: None.
        val_transform (callable, optional): Transformation to apply to the validation data. Default: None.
        test_transform (callable, optional): Transformation to apply to the test data. Default: None.
        train_size (float): Size of the training set. Default: 0.8.
        val_size (float): Size of the validation set. Default: 0.1.
        load_in_memory (bool, optional): Whether to load the data into memory. Default: False.
        psnr_threshold (int): PSNR threshold to filter the data. Default: 30.
    """

    def __init__(
        self,
        input_path,
        target_path,
        mapping_file,
        batch_size=32,
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        test_transform: Callable | None = None,
        train_size=0.8,
        val_size=0.1,
        load_in_memory=False,
    ):
        input_files, target_files = self.read_mapping_files(mapping_file)

        super().__init__(
            input_path,
            target_path,
            input_files,
            target_files,
            batch_size,
            train_transform,
            val_transform,
            test_transform,
            train_size,
            val_size,
            load_in_memory,
        )

