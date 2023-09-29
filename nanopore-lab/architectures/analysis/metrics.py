"""
Script for computing model metrics.
"""
import pathlib
import pickle

import torch

from torch.utils.data import DataLoader, Dataset

from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights



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
        return image, label

if __name__ == "__main__":

    # Prepare data
    test_ds = TestDataset(dataset_dir="/data/jhossbach/Image_Dataset/dataset_all")
    test_dl = DataLoader(test_ds, batch_size=200, shuffle=False)

    # Load model
    model = resnext101_64x4d()
    layer = model.conv1
    new_layer = torch.nn.Conv2d(in_channels=2,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    bias=layer.bias)
    model.conv1 = new_layer
    model.fc = torch.nn.Linear(model.fc.in_features, 42)

    model.

    predictions


