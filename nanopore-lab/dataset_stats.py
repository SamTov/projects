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

import pathlib
from typing import Callable, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as functional
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import matplotlib.pyplot as plt



class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: Literal["train", "validation", "test"],
        padding: int = 927,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """Image dataset.

        Padding is applied to the right side of the image (at the end of the event).
        The image is transformed to a torch tensor automatically, so no need to transform it by hand.

        Parameters
        ----------
        dataset_dir : str
            Parent directory of the dataset.
        subset : Literal['train', 'validation', 'test']
            Subset used for the dataset.
        padding : int
            Size in x-axis the image should have. Must be at least 927 (Maximum image size present)
        transform, target_transform : Optional[Callable]
            Optional transforms.
        """
        if padding < 927:
            raise ValueError("Padded images must be at least 927 in length.")

        self.transform = transform
        self.target_transform = target_transform
        self.ds_dir = pathlib.Path(dataset_dir) / subset
        self.labels = np.load(self.ds_dir / "labels.npy")
        self.padding = padding

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = np.load(self.ds_dir / f"{idx}.npy")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        # Pad to equal length
        # Calculate the total padding needed
        total_padding = self.padding - img.shape[-1]

        # Ensure the total padding is non-negative
        total_padding = max(total_padding, 0)

        # Divide the padding to apply it evenly on both sides
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        img = functional.pad(
            torch.from_numpy(img),
            pad=(left_padding, right_padding),
            mode="constant",
            value=0,
        )
        return img


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


    # # Usage of the dataset
    # transform = transforms.Compose([
    #     # transforms.ToTensor(),
    #     transforms.Normalize(mean, std),
    #     transforms.Resize((128, 128)),
    #     # transforms.ToTensor(),
    # ])

    train_ds = ImageDataset(
        dataset_dir="/work/jhossbach/ImageDatasets/dataset_all_max_length_927",
        subset="train",
    )
    
    dataloader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=1)

    # image = train_ds.__getitem__(10)

    # plt.imshow(np.linalg.norm(np.moveaxis(image.numpy(), 0, -1), axis=-1))
    # plt.savefig("normed_128.png")

    # print(image.shape)
    mean, std = get_mean_and_std(dataloader)
    print(f"Mean: {mean}")
    print(f"Std: {std}")
