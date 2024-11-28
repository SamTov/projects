# System imports and GPU settings
import pathlib
import pickle
import polars
import glob

# Torch imports
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchmetrics
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import v2

# Data management
import numpy as np

# Model imports
from torchvision.models import resnet18, ResNet18_Weights
from dataloaders import split_data, WaveletDataset, DataModule, LitResModel

# Lightning imports
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import StochasticWeightAveraging
import torchvision.transforms as transforms

torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":

    # Hyperparameters (to be tuned)
    hyperparameters = {
        "batch_size": 50,
        "lr": 1e-5,
        # "momentum": 0.9,
        "seed": 38,
        "num_target_classes": 42,
        "max_epochs": 5000,
        "T_max": 1000,
    }
    image_size=244

    # Create default resenet and change first layer to handle
    # new data input.
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
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
        experiment_name=f"resnet18-pretrained-{image_size}", 
        save_dir=f"/work/stovey/sandro/resnext/resnet18-pretrained-{image_size}/mlruns/"
    )

    # Checkpointing for training state.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="ptl/validation_accuracy",
        dirpath=f"/work/stovey/sandro/resnext/resnet18-pretrained-{image_size}/ckpts/",
        filename="resnet18-pretrained-{epoch:02d}-{ptl/validation_loss:.2f}",
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
        default_root_dir=f"/work/stovey/sandro/resnext/resnet18-pretrained-{image_size}/mlruns/",
        enable_progress_bar=True,
        sync_batchnorm=True,
    )

    # Define the normalization transform
    seed = 1
    split = [0.7, 0.15, 0.15]
    use_unlabeled = True

    ### wavelet data ###
    # labeled data
    df_wavelet_labeled = polars.read_json('/work/skuppel/ImageDatasets/dataset_labeled_ssqueezepy_gmw_gamma3_beta60_resized224/metadata.json')
    train, val , test, mean, std = split_data(df_wavelet_labeled, split, seed=seed, calc_mean=True)
    
    ### define transforms ###
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x)),
        transforms.Normalize(mean, std),
    ])

    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x)),
        transforms.Normalize(mean, std),
    ])  
    
    ### create datasets ###
    train_ds_wavelet = WaveletDataset(
        '/work/skuppel/ImageDatasets/dataset_labeled_ssqueezepy_gmw_gamma3_beta60_resized224/',
        df_wavelet_labeled,
        train,
        transform=train_transform,
        numpy=True
    )
    val_ds = WaveletDataset(
        '/work/skuppel/ImageDatasets/dataset_labeled_ssqueezepy_gmw_gamma3_beta60_resized224/',
        df_wavelet_labeled,
        val,
        transform=transform,
        numpy=True
    )
    test_ds = WaveletDataset(
        '/work/skuppel/ImageDatasets/dataset_labeled_ssqueezepy_gmw_gamma3_beta60_resized224/',
        df_wavelet_labeled,
        test,
        transform=transform,
        numpy=True
    )

    # Use the custom datamodule
    datamodule = DataModule(
        batch_size=hyperparameters["batch_size"],
        train_ds=train_ds_wavelet,
        test_ds=test_ds
    )

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=hyperparameters["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=hyperparameters["T_max"])

    # Lightning model definition
    lit_model = LitResModel(
        hyperparameters=hyperparameters,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # ckpts = glob.glob(f"/work/stovey/sandro/resnext/resnet18-pretrained-{image_size}/ckpts/*")
    # epochs = [int(ckpt.split("/")[-1].split("-")[2].split("=")[-1]) for ckpt in ckpts]
    # max_epoch_location = epochs.index(max(epochs))
    # ckpt_path = glob.glob(f"{ckpts[max_epoch_location]}/*")[0]

    # Start training
    trainer.fit(lit_model, datamodule=datamodule, ckpt_path=None)
