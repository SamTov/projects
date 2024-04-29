import pathlib
import pickle

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torchvision import datasets, transforms

from rich.progress import track
import matplotlib.pyplot as plt
import numpy as np


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
        return image


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
        return image


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for data in track(dataloader):
        # Mean over batch, height, and width, but not over the channel
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std


if __name__ == "__main__":

    mean = torch.tensor([3.6632e-05, 6.1436e-06])
    std = torch.tensor([0.0169, 0.0168])

    # Usage of the dataset
    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # transforms.Resize((128, 128)),
        # transforms.ToTensor(),
    ])

    train_ds = TrainingDataset(
            dataset_dir="/work/jhossbach/Image_Dataset_old/dataset_all",
            transform=transform,
            target_transform=None,
        )
    
    dataloader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=1)

    image = train_ds.__getitem__(10)

    # plt.imshow(np.linalg.norm(np.moveaxis(image.numpy(), 0, -1), axis=-1))
    # plt.savefig("normed_128.png")

    print(image.shape)
    # mean, std = get_mean_and_std(dataloader)
    # print(f"Mean: {mean}")
    # print(f"Std: {std}")
