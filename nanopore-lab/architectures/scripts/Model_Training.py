#!/usr/bin/env python
# coding: utf-8

# # ResNet Training Script
# 
# This script uses the created images at the end of the pre-processing pipeline.
# It contains
#  - The definition of the torch dataset and dataloader
#  - The definition of the model used
#  - The Training of the model using Pytorch-Lightning

# In[ ]:

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


import pathlib
import pickle

import torch
import lightning as L
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import StochasticWeightAveraging


torch.set_float32_matmul_precision('medium')

# In[ ]:


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
            #print(label)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# In[ ]:


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
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# In[ ]:


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


# ## Torch model, Lightning model definition

# In[ ]:

import lightning.pytorch as pl
import torch.nn.functional as F
import torchmetrics
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights


# In[ ]:


class LitResModel(pl.LightningModule):
    def __init__(self, hyperparameters, model, optimizer, scheduler):
        super().__init__()

        self.lr = hyperparameters["lr"]
        self.mom = hyperparameters.get("momentum", None)
        pl.seed_everything(hyperparameters["seed"])

        self.opt = optimizer
        self.scheduler = scheduler

        # Needed for manual optimization, see https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
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
        out = self.model.forward(x)
        return out

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        #class_samples = torch.Tensor([labels[labels == item].shape[0] for item in labels.unique()])
        #class_weights = class_samples.shape[0] / class_samples

        preds = self.forward(inputs)
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
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.mom)
        #return [self.opt], [{"scheduler": self.scheduler, "interval": "epoch"}]


# ## Parameter definition, Initialization

# In[ ]:


from pytorch_lightning.accelerators import find_usable_cuda_devices
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


# In[ ]:


# Hyperparameters (to be tuned)
hyperparameters = {
    "batch_size": 50,
    "lr": 1e-3,
    # "momentum": 0.9,
    "seed": 38,
    "num_target_classes": 5,
    "max_epochs": 50,
    "T_max": 1000,
}


# In[ ]:


# Define pre-trained model
#model = resnet101(weights="IMAGENET1K_V2")
#model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
#model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights)
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
# In[ ]:

model.fc = torch.nn.Linear(model.fc.in_features, 5)

# Define logger (insert favorite logger)
logger = pl.loggers.MLFlowLogger(
    experiment_name="dataset_L1AS5_L2AS5_L3AS5_L5AS5_L6AS5",
)


# In[ ]:


# Lightning trainer definition
trainer = pl.Trainer(
    accelerator="gpu",
    logger=logger,
    accumulate_grad_batches=5,
    callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
    #auto_scale_batch_size='binsearch',
    devices="1", #find_usable_cuda_devices(1),
    strategy="ddp",
    max_epochs=hyperparameters["max_epochs"],
    log_every_n_steps=1,
    enable_progress_bar=True,
    sync_batchnorm=True,
)

#tuner = Tuner(trainer)
#tuner.scale_batch_size(model, mode="binsearch")

# In[ ]:

# Use the custom datamodule
datamodule = DataModule(
    data_dir="/data/jhossbach/Image_Dataset/dataset_L1AS5_L2AS5_L3AS5_L5AS5_L6AS5",
    batch_size=hyperparameters["batch_size"],
)


# In[ ]:


# Optimizer and scheduler
optimizer = Adam(model.parameters(), lr=hyperparameters["lr"])
scheduler = CosineAnnealingLR(optimizer, T_max=hyperparameters["T_max"])


# In[ ]:


# Lightning model definition
lit_model = LitResModel(
    hyperparameters=hyperparameters,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
)


# In[ ]:

# Optimizer learning rate before training the model.
# Create a Tuner
tuner = Tuner(trainer)

# finds learning rate automatically
# sets hparams.lr or hparams.learning_rate to that learning rate
tuner.lr_find(lit_model, datamodule)

# Start training
trainer.fit(lit_model, datamodule=datamodule)


# In[ ]:




