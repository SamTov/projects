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
import glob
import numpy as np
# Torch imports
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchmetrics
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import v2

# Model imports
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights

# Lightning imports
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import StochasticWeightAveraging
from collections import OrderedDict
from torchmetrics.classification import MulticlassConfusionMatrix

torch.set_float32_matmul_precision('medium')
import matplotlib.pyplot as plt

from rich.progress import track

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
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=4,
        )

## Torch model, Lightning model definition

class LitResModel(pl.LightningModule):
    def __init__(self, hyperparameters, model, optimizer, scheduler):
        super().__init__()

        self.lr = hyperparameters["lr"]
        self.mom = hyperparameters.get("momentum", None)
        pl.seed_everything(hyperparameters["seed"])

        self.opt = optimizer
        self.scheduler = scheduler

        # self.automatic_optimization = False

        self.model = model

        # Accuracy measures
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=hyperparameters["num_target_classes"]
        )
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=hyperparameters["num_target_classes"]
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        preds = self(inputs)
        loss = F.cross_entropy(preds, labels)
        acc = self.train_acc(preds.argmax(1), labels)

        self.log("ptl/train_loss", loss, sync_dist=True)
        self.log(
            "ptl/train_accuracy",
            acc,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self, *args):
        for indx, lr in enumerate(self.scheduler.get_last_lr()):
            self.log(f"ptl/learning_rate_{indx}", lr)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        preds = self.forward(inputs)
        loss = F.cross_entropy(preds, labels)
        acc = self.valid_acc(preds.argmax(1), labels)

        self.log("ptl/validation_loss", loss, sync_dist=True)
        self.log(
            "ptl/validation_accuracy",
            acc,
            sync_dist=True,
        )
        return {"val_loss": loss, "val_accuracy": acc}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels = batch
        with torch.no_grad():
            preds = self.forward(inputs)
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

## Parameter definition, Initialization
image_size = 244

# Hyperparameters (to be tuned)
hyperparameters = {
    "batch_size": 500,
    "lr": 1e-4,
    # "momentum": 0.9,
    "seed": 38,
    "num_target_classes": 42,
    "max_epochs": 5000,
    "T_max": 1000,
}

# Create default resenet and change first layer to handle
# new data input.
model = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.DEFAULT)
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

ckpt = torch.load(
    "/work/stovey/validation_loss=0.63.ckpt"
)

new_state_dict = OrderedDict()
for k, v in ckpt["state_dict"].items():
    name = k[6:] # remove `model.`
    new_state_dict[name] = v

# load params
model.load_state_dict(new_state_dict)
model.float()
# Define the normalization transform
mean = torch.tensor([3.6632e-05, 6.1436e-06])
std = torch.tensor([0.0169, 0.0168])
train_transform = v2.Compose(
    [
        v2.Normalize(mean, std),
    ]
)
test_transform = v2.Compose(
    [
        v2.Normalize(mean, std),
    ]
)

# Use the custom datamodule
datamodule = DataModule(
    data_dir="/work/jhossbach/Image_Dataset_old/dataset_all",
    batch_size=hyperparameters["batch_size"],
    train_transform=None,
    test_transform=None
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

lit_model.eval()

val_dl = datamodule.val_dataloader()
tr_dl = datamodule.train_dataloader()

targets = []
predictions = []
accuracy = []

for item in track(val_dl):
    inputs, labels = item
    y_pred = lit_model(inputs).argmax(1)

    accuracy.append(torch.mean((y_pred == labels).float()).numpy())
    targets.append(labels)
    predictions.append(y_pred)

targets = torch.cat(targets, dim=0)
predictions = torch.cat(predictions, dim=0)

print(f"Accuracy: {torch.mean((predictions == targets).float()).numpy()}")
metric = MulticlassConfusionMatrix(num_classes=42)

cm = metric(predictions, targets)

print(cm)
np.save("cm.npy", cm)

plt.imshow(cm)
plt.savefig("cm.png")