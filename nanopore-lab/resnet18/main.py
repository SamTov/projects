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
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchmetrics
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

# Model imports
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights

# Lightning imports
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import StochasticWeightAveraging

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


class DataModule(L.LightningDataModule):
    def __init__(
        self, batch_size: int, data_dir, transform=None, target_transform=None
    ):
        super().__init__()
        self.batch_size = batch_size

        self.train_ds = TrainingDataset(
            dataset_dir=data_dir,
            transform=transform,
            target_transform=target_transform,
        )
        self.val_ds = ValidationDataset(
            dataset_dir=data_dir,
            transform=transform,
            target_transform=target_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=4,
        )

## Torch model, Lightning model definition



## Parameter definition, Initialization
image_size = 4

# Hyperparameters (to be tuned)
hyperparameters = {
    "batch_size": 50,
    "lr": 1e-4,
    # "momentum": 0.9,
    "seed": 38,
    "num_target_classes": 42,
    "max_epochs": 5000,
    "T_max": 1000,
}

# Create default resenet and change first layer to handle
# new data input.
model = resnext101_64x4d()
# Change the first layer
layer = model.conv1
new_layer = torch.nn.Conv2d(in_channels=2,
                  out_channels=layer.out_channels,
                  kernel_size=layer.kernel_size,
                  stride=layer.stride,
                  padding=layer.padding,
                  bias=layer.bias)

model.conv1 = new_layer
model.fc = torch.nn.Linear(model.fc.in_features, 42)


# Define logger (insert favorite logger)
logger = pl.loggers.MLFlowLogger(
    experiment_name=f"resnext101-{image_size}", 
    save_dir=f"/work/stovey/resnext/resnext101-{image_size}/mlruns/"
)

# Checkpointing for training state.
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="ptl/validation_accuracy",
    dirpath=f"/work/stovey/resnext/resnext101-{image_size}/ckpts/",
    filename="resnext101-{epoch:02d}-{ptl/validation_loss:.2f}",
    save_top_k=2,
    mode="max",
)

# Lightning trainer definition
trainer = pl.Trainer(
    accelerator="gpu",
    logger=logger,
    accumulate_grad_batches=100,
    callbacks=[StochasticWeightAveraging(swa_lrs=1e-2), checkpoint_callback],
    devices="auto",
    strategy="ddp",
    max_epochs=hyperparameters["max_epochs"],
    log_every_n_steps=10,
    default_root_dir=f"/work/stovey/resnext/resnext101-{image_size}/mlruns/",
    enable_progress_bar=True,
    sync_batchnorm=True,
)

# Define the normalization transform
mean = torch.tensor([3.6632e-05, 6.1436e-06])
std = torch.tensor([0.0169, 0.0168])
transform = transforms.Compose(
    [
        transforms.Normalize(mean, std),
        transforms.Resize((4, 4)),
    ]
)

# Use the custom datamodule
datamodule = DataModule(
    data_dir="/work/jhossbach/Image_Dataset_old/dataset_all",
    batch_size=hyperparameters["batch_size"],
    transform=transform,
)

# Optimizer and scheduler
optimizer = Adam(model.parameters(), lr=hyperparameters["lr"], weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=hyperparameters["T_max"])

# Lightning model definition
lit_model = LitResModel(
    hyperparameters=hyperparameters,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
)

# Start training
trainer.fit(lit_model, datamodule=datamodule)
