from typing import Callable, Optional

import numpy as np
import polars
import torch
import torchvision.transforms as transforms
import zarr
from torch.utils.data import Dataset
import lightning as L
import lightning.pytorch as pl
import torchmetrics
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


def split_data(
    df: polars.DataFrame,
    fraction: tuple[float, float, float],
    seed: int,
    keep_imbalance: bool = True,
    calc_mean: Optional[bool] = False,
    max_length: Optional[int] = None,
):
    """Construct the dataset.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame containing the dataset.
    max_length : int, optional
        Maximum length of events. others are discarded.
    """
    if "index" not in df.columns:
        df = df.with_row_index()
    if max_length:
        if "length" in df.columns:
            df = df.filter(polars.col("length") <= max_length)
        if "edge" in df.columns:
            df = df.with_columns(
                (polars.col("edge").list.last() - polars.col("edge").list.first()).alias("length")
            )
            df = df.filter(polars.col("length") <= max_length)
        else:
            raise ValueError("No length or edge column found in the dataset")

    if not sum(fraction) == 1:
        raise ValueError("Fractions do not sum up to 1!")
    if calc_mean:
        if "mean" not in df.columns:
            raise ValueError("Mean not found in the dataset but calc_mean is set to True.")
        if "std" not in df.columns:
            raise ValueError(
                "Standard deviation not found in the dataset but calc_mean is set to True."
            )
    train_indxs, val_indxs, test_indxs = [], [], []
    if keep_imbalance:
        for grouped_df in df.partition_by("label"):
            # Compute fraction for each df as grouped by label
            num_per_fraction = np.cumsum(
                [np.floor(grouped_df.height * _fraction).astype(int) for _fraction in fraction]
            )
            # Shuffle
            indxs = grouped_df["index"].shuffle(seed=seed).to_list()
            # Add to index list
            train_indxs.extend(indxs[: num_per_fraction[0]])
            val_indxs.extend(indxs[num_per_fraction[0] : num_per_fraction[1]])
            test_indxs.extend(indxs[num_per_fraction[1] : num_per_fraction[2]])
    else:
        # Compute fraction for whole df
        num_per_fraction = np.cumsum(
            [np.floor(df.height * _fraction).astype(int).astype(int) for _fraction in fraction]
        )
        # Shuffle
        indxs = df["index"].shuffle(seed=seed).to_list()
        # Add to index list
        train_indxs.extend(indxs[: num_per_fraction[0]])
        val_indxs.extend(indxs[num_per_fraction[0] : num_per_fraction[1]])
        test_indxs.extend(indxs[num_per_fraction[1] : num_per_fraction[2]])
    if calc_mean:
        train_df = df.filter(polars.col("index").is_in(train_indxs))
        mean = np.mean(train_df["mean"].to_numpy(), axis=0)
        std = np.mean(train_df["std"].to_numpy(), axis=0)
        return train_indxs, val_indxs, test_indxs, mean, std
    return train_indxs, val_indxs, test_indxs


class WaveletDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        df: polars.DataFrame,
        index_list: Optional[list[int]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        numpy = False
    ) -> None:
        """Dataset for reading the wavelet images.

        Parameters
        ----------
        image_dir : str
            Directory containing the wavelet images.
        df : polars.DataFrame
            Dataframe containing the index and label.
        index_list : list[int], optional
            List of indices to use from the dataframe.
            If None, all indices will be used.
        transform : Callable, optional
            Optional transforms to perform on the image.
        target_transform : Callable, optional
        """
        self.numpy = numpy
        self.image_dir = image_dir
        self.df = df
        if index_list:
            self.index_list = index_list
        else:
            self.index_list = list(range(self.df.height))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.index_list)

    def __getitem__(self, idx: int):
        """Fetch Wavelet and its according label.

        Parameters
        ----------
        idx : int
            Index to fetch from the dataframe.

        Returns
        -------
        img: np.ndarray
            The wavelet image. If a `transform` is defined, returns
            the output of the `transform` function.
        label: int
            The label. If a `target_transform` is defined, returns
            the output of the `target_transform` function
        """
        if self.numpy:
            img = np.load(f"{self.image_dir}/{self.index_list[idx]}.npy")
        else:
            img = zarr.load(f"{self.image_dir}/{self.index_list[idx]}")
        label = self.df["label"][self.index_list[idx]]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


class DataModule(L.LightningDataModule):
    def __init__(
        self, 
        batch_size: int, 
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
            task="multiclass", num_classes=hyperparameters["num_target_classes"], average="weighted"
        )
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=hyperparameters["num_target_classes"], average="weighted"
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