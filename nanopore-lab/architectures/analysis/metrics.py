"""
Script for computing model metrics.
"""
import pathlib
import pickle
from collections import OrderedDict

# Torch imports
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Model imports
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights

# Lightning imports
import lightning as L
from lightning.callbacks import StochasticWeightAveraging
import matplotlib.pyplot as plt
torch.set_printoptions(profile="full")
torch.set_float32_matmul_precision('medium')

## Torch model, Lightning model definition

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


## Parameter definition, Initialization

# Hyperparameters (to be tuned)
hyperparameters = {
    "batch_size": 10,
    "lr": 1e-3,
    # "momentum": 0.9,
    "seed": 38,
    "num_target_classes": 42,
    "max_epochs": 100,
    "T_max": 1000,
}


class TestDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.ds_dir = pathlib.Path(dataset_dir)

        with open(self.ds_dir / "metadata.pk", "rb") as f:
            self.metadata = pickle.load(f)

    def __len__(self):
        return self.metadata["val_ds_size"]

    def __getitem__(self, idx):
        with open(self.ds_dir / f"test/{idx}.pk", "rb") as f:
            label, image = pickle.load(f)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return torch.Tensor(image), label
    

if __name__ == "__main__":

    # Prepare data
    test_ds = TestDataset(dataset_dir="/data/jhossbach/Image_Dataset/dataset_all")
    test_dl = DataLoader(test_ds, batch_size=20, shuffle=True)

    # Load model
    model = resnext101_64x4d()
    # Change the first layer
    layer = model.conv1
    new_layer = torch.nn.Conv2d(
        in_channels=2,
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        bias=layer.bias
    )

    model.conv1 = new_layer
    model.fc = torch.nn.Linear(model.fc.in_features, 42)

    state_dict = torch.load(
        "/data/stovey/ResNext-Models/ResNext101_64x4d-pretrained/ckpts/ResNext101_64x4d-pretrained-epoch=28-ptl/validation_loss=0.96.ckpt"
        )

    new_state_dict = OrderedDict()
    for k, v in state_dict["state_dict"].items():
        name = k[6:] # remove `module.`
        new_state_dict[name] = v

    # load params
    model.load_state_dict(new_state_dict)
    model.float()

    metric = MulticlassConfusionMatrix(num_classes=42)
    for i, item in enumerate(test_dl):
        image, labels = item
        predictions = model(image)

        if i == 0:
            preds = F.softmax(predictions, dim=1)
            labs = F.one_hot(labels, num_classes=42)
            c_matrix = metric(
                    preds, labs
                )
        else:
            preds = F.softmax(predictions, dim=1)
            labs = F.one_hot(labels, num_classes=42)
            c_matrix += metric(
                    preds, labs
                )
            
    np.save("confusion_matrix.npy", c_matrix.numpy())