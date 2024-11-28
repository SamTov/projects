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

# Torch imports
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchmetrics
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import v2
import numpy as np
# Model imports
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights

# Lightning imports
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import StochasticWeightAveraging

import torchvision.transforms as transforms

torch.set_float32_matmul_precision('medium')


# Build datasets and dataloaders
class ImageDataset(Dataset):
    def __init__(self, dataset_dir, subset, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.ds_dir = pathlib.Path(dataset_dir) / subset
        self.labels = np.load(self.ds_dir / "labels.npy")


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.Tensor(np.load(self.ds_dir / f"{idx}.npy"))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class DataModule(L.LightningDataModule):
    def __init__(
        self, 
        batch_size: int, 
        data_dir, 
        train_ds=None, 
        test_ds = None, 
        target_transform=None
    ):
        super().__init__()
        self.batch_size = batch_size

        self.train_ds = train_ds
        self.val_ds = test_ds

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
    "batch_size": 50,
    "lr": 8e-5,
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


# Define logger (insert favorite logger)
logger = pl.loggers.MLFlowLogger(
    experiment_name=f"resnext101-pretrained-{image_size}", 
    save_dir=f"/work/stovey/sandro/resnext/resnext101-pretrained-{image_size}/mlruns/"
)

# Checkpointing for training state.
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="ptl/validation_accuracy",
    dirpath=f"/work/stovey/sandro/resnext/resnext101-pretrained-{image_size}/ckpts/",
    filename="resnext101-pretrained-{epoch:02d}-{ptl/validation_loss:.2f}",
    save_top_k=2,
    mode="max",
)

# Lightning trainer definition
trainer = pl.Trainer(
    accelerator="gpu",
    logger=logger,
    accumulate_grad_batches=16,
    callbacks=[StochasticWeightAveraging(swa_lrs=1e-2), checkpoint_callback],
    devices="auto",
    strategy="ddp",
    max_epochs=hyperparameters["max_epochs"],
    log_every_n_steps=10,
    default_root_dir=f"/work/stovey/sandro/resnext/resnext101-pretrained-{image_size}/mlruns/",
    enable_progress_bar=True,
    sync_batchnorm=True,
)

# Define the normalization transform
size = 224

mean = torch.tensor([3.5366e-05, 2.3211e-05])
std = torch.tensor([0.0110, 0.0109])

transform = transforms.Compose([
      transforms.Resize((size, size)),
      transforms.Normalize(mean, std),
])
print('loading data')
# dataset max length 927 
train_dataset = ImageDataset(
   '/work/skuppel/dataset_all_max_length_927_image',
    subset='train',
    transform = transform
)

val_dataset = ImageDataset(
    '/work/skuppel/dataset_all_max_length_927_image',
    subset='validation',
    transform = transform
)

test_dataset = ImageDataset(
    '/work/skuppel/dataset_all_max_length_927_image',
    subset='test',
    transform = transform
)

# Use the custom datamodule
datamodule = DataModule(
    data_dir="/work/jhossbach/Image_Dataset_old/dataset_all",
    batch_size=hyperparameters["batch_size"],
    train_ds=train_dataset,
    test_ds=val_dataset
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

# ckpts = glob.glob(f"/work/stovey/resnext/resnext101-pretrained-{image_size}/ckpts/*")
# epochs = [int(ckpt.split("/")[-1].split("-")[1].split("=")[-1]) for ckpt in ckpts]
# max_epoch_location = epochs.index(max(epochs))
# ckpt_path = glob.glob(f"{ckpts[max_epoch_location]}/*")[0]

# Start training
trainer.fit(lit_model, datamodule=datamodule)
