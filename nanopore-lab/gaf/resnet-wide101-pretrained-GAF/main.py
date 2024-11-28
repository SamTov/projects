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
import numpy as np
import glob

# Torch imports
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchmetrics
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import v2
import torch.nn.functional as functional
from typing import Callable, Literal, Optional

# Model imports
from torchvision.models import wide_resnet101_2, Wide_ResNet101_2_Weights

# Lightning imports
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import StochasticWeightAveraging

torch.set_float32_matmul_precision('medium')

class GAFDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: Literal["train", "validation", "test"],
        use_padding: bool = True,
        padding: int = 927,
        symmetric_padding: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """Image dataset.

        Padding is applied to the right side of the image (at the beginning of the event).
        The image is transformed to a torch tensor automatically, so no need to transform it by hand.

        Parameters
        ----------
        dataset_dir : str
            Parent directory of the dataset.
        subset : Literal['train', 'validation', 'test']
            Subset used for the dataset.
        use_padding : boolean, default False
            Whether to add padding to the image.
        padding : int
            Padding to apply on x and y sides. Must be at least 927. Only used when use_padding if True.
        symmetric_padding : bool, default False
            Whether to add padding symmetrically on sides of image.
        transform, target_transform : Optional[Callable]
            Optional transforms.
        """
        if use_padding:
            if padding < 927:
                raise ValueError("Padded images must be at least 927 in length.")

        self.transform = transform
        self.target_transform = target_transform
        self.ds_dir = pathlib.Path(dataset_dir) / subset
        self.labels = np.load(self.ds_dir / "labels.npy")
        self.use_padding = use_padding
        self.padding = padding
        self.symmetric_padding = symmetric_padding

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = np.load(self.ds_dir / f"{idx}.npy")
        label = self.labels[idx]

        img = torch.from_numpy(img)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        # Pad to equal length
        if self.use_padding:
            if self.symmetric_padding:
                left_pad = (self.padding - img.shape[-1]) // 2
                right_pad = self.padding - img.shape[-1] - left_pad
                pad = (left_pad, right_pad, left_pad, right_pad)
            else:
                pad = (self.padding - img.shape[-1], 0, self.padding - img.shape[-2], 0)
            img = functional.pad(
                img,
                pad=pad,
                mode="constant",
                value=0,
            )
        return img, label

class DataModule(L.LightningDataModule):
    def __init__(
        self, batch_size: int, data_dir, train_transform=None, test_transform = None, target_transform=None
    ):
        super().__init__()
        self.batch_size = batch_size

        self.train_ds = GAFDataset(
            dataset_dir=data_dir,
            subset="train",
            transform=train_transform,
            target_transform=target_transform,
        )
        self.val_ds = GAFDataset(
            dataset_dir=data_dir,
            subset="validation",
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
image_size = "GAF"

# Hyperparameters (to be tuned)
hyperparameters = {
    "batch_size": 5,
    "lr": 1e-4,
    # "momentum": 0.9,
    "seed": 38,
    "num_target_classes": 42,
    "max_epochs": 5000,
    "T_max": 1000,
}

# Create default resenet and change first layer to handle
# new data input.
model = wide_resnet101_2(weights=Wide_ResNet101_2_Weights.DEFAULT)

model.fc = torch.nn.Linear(model.fc.in_features, 42)


# Define logger (insert favorite logger)
logger = pl.loggers.MLFlowLogger(
    experiment_name=f"wide_resnet101_2-pretrained-{image_size}", 
    save_dir=f"/work/stovey/resnext/wide_resnet101_2-pretrained-{image_size}/mlruns/"
)

# Checkpointing for training state.
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="ptl/validation_accuracy",
    dirpath=f"/work/stovey/resnext/wide_resnet101_2-pretrained-{image_size}/ckpts/",
    filename="wide_resnet101_2-pretrained-{epoch:02d}-{ptl/validation_loss:.2f}",
    save_top_k=2,
    mode="max",
)

# Lightning trainer definition
trainer = pl.Trainer(
    accelerator="gpu",
    logger=logger,
    num_nodes=2,
    accumulate_grad_batches=1000,
    callbacks=[StochasticWeightAveraging(swa_lrs=1e-2), checkpoint_callback],
    devices=-1,
    strategy="ddp",
    max_epochs=hyperparameters["max_epochs"],
    log_every_n_steps=10,
    default_root_dir=f"/work/stovey/resnext/wide_resnet101_2-pretrained-{image_size}/mlruns/",
    enable_progress_bar=True,
    sync_batchnorm=True,
)

transform = v2.Compose(
    [
        # v2.RandomErasing(0.1),
        # v2.RandomHorizontalFlip(0.2),
        # v2.RandomVerticalFlip(0.2),
        v2.Resize((244, 244)),
    ]
)

# Use the custom datamodule
datamodule = DataModule(
    data_dir="/work/jhossbach/dataset_gaf_max_length_927",
    batch_size=hyperparameters["batch_size"],
    train_transform=transform,
    test_transform=transform
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

# ckpts = glob.glob(f"/work/stovey/resnext/wide_resnet101_2-pretrained-{image_size}/ckpts/*")
# epochs = [int(ckpt.split("/")[-1].split("-")[1].split("=")[-1]) for ckpt in ckpts]
# max_epoch_location = epochs.index(max(epochs))
# ckpt_path = glob.glob(f"{ckpts[max_epoch_location]}/*")[0]

# Start training
trainer.fit(lit_model, datamodule=datamodule)
