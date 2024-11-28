import torch
import torch.nn.functional as F
import torchmetrics
import lightning as L
import lightning.pytorch as pl

from dataclasses import dataclass
import numpy as np


@dataclass
class Parameters:
    storage_path: str
    experiment_name: str
    image_size: int
    epochs: int
    learning_rate: float
    batch_size: int
    batch_accumulation: int


class LitResModel(pl.LightningModule):
    def __init__(self, model, optimizer, loss_module, scheduler=None):
        super().__init__()

        pl.seed_everything(np.random.randint(324235))

        self.opt = optimizer
        self.sched = scheduler
        self.model = model
        self.loss = loss_module

        # Accuracy measures
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=42
        )
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=42
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        # Compute latent space representations
        latents = self.model.latent_space(inputs)

        # Forward pass
        preds = self(inputs)

        # Compute losses
        total_loss, contrastive_loss, supervised_loss = self.loss(latents, labels, preds)
        acc = self.train_acc(preds.argmax(1), labels)

        self.log("ptl/train_loss", total_loss, sync_dist=True)
        self.log("ptl/train_contrastive_loss", contrastive_loss, sync_dist=True)
        self.log("ptl/train_supervised_loss", supervised_loss, sync_dist=True)
        self.log("ptl/train_accuracy", acc, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        preds = self.forward(inputs)
        loss = F.cross_entropy(preds, labels)
        acc = self.valid_acc(preds.argmax(1), labels)

        self.log("ptl/validation_loss", loss, sync_dist=True)
        self.log(
            "ptl/validation_accuracy", acc, sync_dist=True,
        )
        return {"val_loss": loss, "val_accuracy": acc}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels = batch
        with torch.no_grad():
            preds = self.forward(inputs)
        return preds
    
    def configure_optimizers(self):
        return {"optimizer": self.opt, "lr_scheduler": self.sched}
