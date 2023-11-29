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


from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

# Model imports
from transformers import (
    ViTForImageClassification, 
    AdamW, 
    ViTImageProcessor, 
    ViTConfig
)

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

        if self.transform:
            image = image.reshape((224, 224, 2))
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
            # print(label)

        if self.transform:
            image = image.reshape((224, 224, 2))
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class DataModule(L.LightningDataModule):
    def __init__(
        self, batch_size: int, data_dir, train_transform=None, val_transform=None, target_transform=None
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
            transform=val_transform,
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

class LitVitModel(pl.LightningModule):

    def __init__(self, hyperparameters):
        super().__init__()

        config = ViTConfig(num_channels=2, num_labels=42)
        # Define the model
        # self.vit = ViTForImageClassification.from_pretrained(
        #     'google/vit-base-patch16-224-in21k',
        #     config=config,
        #     num_labels=42,
        #     )
        self.vit = ViTForImageClassification(config=config)

        self.lr = hyperparameters["lr"]
        self.mom = hyperparameters.get("momentum", None)
        pl.seed_everything(hyperparameters["seed"])

        # Accuracy measures
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=hyperparameters["num_target_classes"]
        )
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=hyperparameters["num_target_classes"]
        )

    def forward(self, x):
        out = self.vit(x)
        return out.logits

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
        return AdamW(self.parameters(), lr=5e-5, weight_decay=1e-5)

# Hyperparameters (to be tuned)
hyperparameters = {
    "batch_size": 20,
    "lr": 1e-3,
    # "momentum": 0.9,
    "seed": 38,
    "num_target_classes": 42,
    "max_epochs": 500,
    "T_max": 1000,
}

# Data transformations
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_mean = processor.image_mean[:2]
image_std = processor.image_std[:2]
size = processor.size["height"]


normalize = Normalize(mean=image_mean, std=image_std)

_train_transforms = Compose(
        [
            ToTensor(),
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            ToTensor(),
            Resize(size),
            CenterCrop(size),
            normalize,
        ]
    )


# Define logger (insert favorite logger)
logger = pl.loggers.MLFlowLogger(
    experiment_name="ResNext101_64x4d-scratch",
    save_dir="/data/stovey/ViT/mlruns/"
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="ptl/validation_accuracy",
    dirpath="/data/stovey/ViT/ckpts/",
    filename="ResNext101_64x4d-pretrained-{epoch:02d}-{ptl/validation_loss:.2f}",
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
    log_every_n_steps=1,
    default_root_dir="/data/stovey/ViT/ckpts/",
    enable_progress_bar=True,
    sync_batchnorm=True,
)

# Use the custom datamodule
datamodule = DataModule(
    data_dir="/data/jhossbach/Image_Dataset/dataset_all",
    batch_size=hyperparameters["batch_size"],
    train_transform=_train_transforms,
    val_transform=_val_transforms
)

# Lightning model definition
lit_model = LitVitModel(
    hyperparameters=hyperparameters,
)

# Optimizer learning rate before training the model.
# Create a Tuner
#tuner = Tuner(trainer)

# # finds learning rate automatically
# # sets hparams.lr or hparams.learning_rate to that learning rate
#tuner.lr_find(lit_model, datamodule)

# Start training
trainer.fit(lit_model, datamodule=datamodule)
