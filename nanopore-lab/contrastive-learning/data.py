# System imports and GPU settings
import pathlib
import pickle

# Torch imports
import torch
from torch.utils.data import DataLoader, Dataset

# Lightning imports
import lightning as pl

torch.set_float32_matmul_precision('medium')


# Build datasets and dataloaders
class TrainingDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.ds_dir = pathlib.Path(dataset_dir)

        with open(self.ds_dir / "metadata.pk", "rb") as f:
            self.metadata = pickle.load(f)


    def __len__(self):
        return self.metadata["train_ds_size"]

    def __getitem__(self, idx):
        with open(self.ds_dir / f"train/{idx}.pk", "rb") as f:
            label, image = pickle.load(f)
            image = torch.Tensor(image)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class ValidationDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.ds_dir = pathlib.Path(dataset_dir)

        with open(self.ds_dir / "metadata.pk", "rb") as f:
            self.metadata = pickle.load(f)

    def __len__(self):
        return self.metadata["val_ds_size"]

    def __getitem__(self, idx):
        with open(self.ds_dir / f"validation/{idx}.pk", "rb") as f:
            label, image = pickle.load(f)
            image = torch.Tensor(image)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class DataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size: int, data_dir, train_transform=None, test_transform = None, target_transform=None
    ):
        super().__init__()
        self.batch_size = batch_size

        self.train_ds = TrainingDataset(
            dataset_dir=data_dir,
            transform=train_transform,
            target_transform=target_transform,
        )
        self.val_ds = ValidationDataset(
            dataset_dir=data_dir,
            transform=test_transform,
            target_transform=target_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=8,
        )