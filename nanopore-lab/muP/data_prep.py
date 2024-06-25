#!/usr/bin/env python
# coding: utf-8

# # ResNet Training Script
# 
# This script uses the created images at the end of the pre-processing pipeline.
# It contains
#  - The definition of the torch dataset and dataloader
#  - The definition of the model used
#  - The Training of the model using Pytorch-Lightning


# System imports and GPU settings
import pathlib
import pickle

# Torch imports
import torch
from torch.utils.data import Dataset

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
            # print(label)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
