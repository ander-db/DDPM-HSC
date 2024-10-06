import pytest
import os
import torch
from torch.utils.data import DataLoader

from src.data_setup.wide_dud_v2 import FITSHSCDataset, Image2ImageHSCDataModule

# Replace this with the path to your actual test mapping directory
TEST_MAPPING_DIR = "./data/mappings/testing_mapping/"


@pytest.fixture
def datamodule():
    return Image2ImageHSCDataModule(
        mapping_dir=TEST_MAPPING_DIR,
        batch_size=4,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        load_in_memory=False,
        num_workers=0,
    )


def test_datamodule_init(datamodule):
    assert datamodule.mapping_dir == TEST_MAPPING_DIR
    assert datamodule.batch_size == 4
    assert len(datamodule.train_files) > 0
    assert len(datamodule.val_files) > 0
    assert len(datamodule.test_files) > 0


def test_read_mapping_file(datamodule):
    train_files = datamodule.read_mapping_file(
        os.path.join(TEST_MAPPING_DIR, "train_files.txt")
    )
    print(train_files)
    print(len(train_files))
    assert len(train_files) > 0
    print(isinstance(train_files, list))
    print(isinstance(train_files[0], tuple))
    print(len(train_files[0]))
    assert all(isinstance(item, tuple) and len(item) == 2 for item in train_files)
    print(all(os.path.exists(item[0]) and os.path.exists(item[1]) for item in train_files))
    assert all(
        os.path.exists(item[0]) and os.path.exists(item[1]) for item in train_files
    )


@pytest.mark.parametrize("stage", ["fit", "test", None])
def test_setup(datamodule, stage):
    datamodule.setup(stage=stage)

    if stage in ["fit", None]:
        assert isinstance(datamodule.train_dataset, FITSHSCDataset)
        assert isinstance(datamodule.val_dataset, FITSHSCDataset)

    if stage in ["test", None]:
        assert isinstance(datamodule.test_dataset, FITSHSCDataset)


def test_train_dataloader(datamodule):
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    assert isinstance(train_loader, DataLoader)
    assert train_loader.batch_size == 4


def test_val_dataloader(datamodule):
    datamodule.setup("fit")
    val_loader = datamodule.val_dataloader()
    assert isinstance(val_loader, DataLoader)
    assert val_loader.batch_size == 4


def test_test_dataloader(datamodule):
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    assert isinstance(test_loader, DataLoader)
    assert test_loader.batch_size == 4


@pytest.mark.parametrize(
    "dataset_attr", ["train_dataset", "val_dataset", "test_dataset"]
)
def test_dataloader_not_setup(datamodule, dataset_attr):
    with pytest.raises(
        ValueError,
        match=f"{dataset_attr.split('_')[0].capitalize()} dataset is not set up",
    ):
        getattr(datamodule, f"{dataset_attr.split('_')[0]}_dataloader")()


def test_dataset_getitem(datamodule):
    datamodule.setup("fit")
    input_tensor, target_tensor = datamodule.train_dataset[0]
    assert isinstance(input_tensor, torch.Tensor)
    assert isinstance(target_tensor, torch.Tensor)
    assert input_tensor.dim() == 3  # Assuming 2D image with channel
    assert target_tensor.dim() == 3


def test_dataloader_iteration(datamodule):
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    assert len(batch) == 2
    inputs, targets = batch
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert inputs.dim() == 4  # Batch of 2D images with channel
    assert targets.dim() == 4
    assert inputs.shape[0] == datamodule.batch_size
    assert targets.shape[0] == datamodule.batch_size
