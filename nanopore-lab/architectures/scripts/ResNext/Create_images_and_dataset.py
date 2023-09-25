#!/usr/bin/env python
# coding: utf-8

# # Image and Pytorch-Dataset creation

# ## Image creation methods
# 
# This part of the notebook uses the pre-labelled data and produces the CWT images.
# 
# One can choose from the available peptides by name ("L1AS4", "L2AS9" e.g.)
# 
# The number of images for each label is set to the smallest number available of the chosen peptides in order to create even sized classes.
# 
# The creation of the dataset structure and choosing the events for the validation and training dataset are split on purpose in order to parallelize the image creation later on.

# In[ ]:


import logging
import pathlib
import pickle
import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from ssqueezepy import ssq_cwt
from torchvision.transforms import Compose, Resize


# In[ ]:


def compute_events_labels(
    peptides: list[str],
    validation_percentage: float,
    peptide_dir: str,
    output_dir: str,
    seed: int = 17,
    num_events: int = None,
):
    """Compute the events and labels.

    Produces the following structure:
    ```
    output_directory
    ```

    Parameters
    ----------
    peptides : list[str]
        Names of the .pk files to use for image generation.
    validation_percentage : float
        Percentage of events to use for validation. Between 0 and 1.
    peptide_dir : str
        Path to the labelled peptide files.
    output_dir : str
        Name of the directory to create subdirectory structure (see above for details).
        Fails if subdirectory with given classes already exists.
    seed : int, default 17
        Random seed to use for selecting the events.
    num_events : int
        Number of events, if none provided the min number of the used peptides is used.
    """
    rng = np.random.default_rng(seed)

    peptide_dir = pathlib.Path(peptide_dir)

    # Create dataset structure
    dataset_path = pathlib.Path(output_dir) / f"dataset_{('_'.join(peptides))}"
    dataset_path.mkdir()

    train_dataset_path = dataset_path / "train"
    val_dataset_path = dataset_path / "validation"
    train_dataset_path.mkdir()
    val_dataset_path.mkdir()

    (train_dataset_path / "events").mkdir()
    (val_dataset_path / "events").mkdir()

    # Get minimum number of samples to avoid imbalanced datasets
    if num_events is None:
        lengths = []
        for peptide in peptides:
            with open(peptide_dir / (peptide + ".pk"), "rb") as f:
                data = pickle.load(f)
                lengths.append(len(data))

        per_class_size = min(lengths)
    else:
        per_class_size = num_events
    print(f"Using {per_class_size} images per class.")

    # Show imbalance of classes
    plt.bar(peptides, lengths)
    plt.xlabel("Peptide")
    plt.ylabel("Number of events")
    plt.title("Number of events for chosen peptides")
    plt.show()

    full_train_label_list = []
    full_train_event_list = []
    full_val_label_list = []
    full_val_event_list = []

    for label, peptide in zip(range(len(peptides)), peptides):
        print(f"Peptide {peptide} with label {label}.")

        with open(peptide_dir / (peptide + ".pk"), "rb") as f:
            data = pickle.load(f)

        # Permute indices and take subsample as decided by max_num_class
        indices = rng.permutation(data.shape[0])[:per_class_size]
        limit = int(validation_percentage * len(indices))
        train_indxs, val_indxs = indices[limit:], indices[:limit]

        full_train_event_list.extend(data[train_indxs])
        full_train_label_list.extend(len(train_indxs) * [label])
        full_val_event_list.extend(data[val_indxs])
        full_val_label_list.extend(len(val_indxs) * [label])

    # Save metadata
    with open(dataset_path / "metadata.pk", "wb") as f:
        dict_ = dict(zip(range(len(peptides)), peptides))
        dict_.update(
            {
                "train_ds_size": len(full_train_label_list),
                "val_ds_size": len(full_val_label_list),
            }
        )
        pickle.dump(dict_, f)

    return (
        dataset_path,
        full_train_event_list,
        full_train_label_list,
        full_val_event_list,
        full_val_label_list,
    )


# In[ ]:


def generate_and_save_img(
    event: np.ndarray,
    label: int,
    index: int,
    data_dir: str,
    size: tuple[int],
    use_abs: bool = False,
    use_ssq: bool = False,
    scales: Union[str, np.ndarray] = "log-piecewise",
    wavelets: Union[tuple, list[tuple]] = ("gmw", {"dtype": "float32"}),
):
    """Generate and save an image from a given event.

    Parameters
    ----------
    event : np.ndarray
    label : int
        Label (class) of the event.
    index : int
        Running index of the event based on the previous dataset selection.
    data_dir : str
        Directory to save image in. Either "training" or "validation".
    size : tuple[int]
        Size of the image to be produced.
    use_abs : bool, default False
        Whether or not to use the absolute value instead of splitting the complex and imaginary part.
    use_ssq : bool, default False
        Whether to squeeze the CWT. Defaults to False
    scales : np.ndarray
        Scales used for the wavelet. Defaults to "log-piecewise".
    wavelets: tuple, list of tuples or wavelets
        What tuple to use. Defaults to ("gmw", {"dtype": "float32"})
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="overflow encountered in scalar power"
        )
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in scalar multiply"
        )

        def gen_img(event, wavelet_tuple):
            # Compute the image, split the real and imaginary part as channels.
            squeezed, img, *_ = ssq_cwt(
                event, scales=scales, wavelet=wavelet_tuple, fs=1e6
            )
            if use_ssq:
                img = squeezed
            if use_abs:
                res = np.abs(img)
                # Add new axis for channels
                res = res[np.newaxis, ...]
            else:
                # Stack imaginary and real part in first axis (channel axis)
                res = np.stack([img.real, img.imag], axis=0)
            return res.astype(np.float32)

        # Resizing
        transform = Compose([torch.from_numpy, Resize(size, antialias=True)])

        # If a list of wavelets is provided, use channel axis with different wavelets
        if isinstance(wavelets, list):
            assert use_abs == True
            imgs = []
            for wavelet_tuple in wavelets:
                img = gen_img(event, wavelet_tuple)
                print(img.shape)
                imgs.append(img)
            img = np.stack(imgs, axis=0)
        else:
            img = gen_img(event, wavelets)
        transformed_img = transform(img).numpy()

        # Sanity check for image
        assert np.isfinite(transformed_img).all()
        if pathlib.Path(data_dir / f"{index}.pk").is_file():
            raise

        # Dump image in index.pk file
        with open(data_dir / f"{index}.pk", "wb") as f:
            pickle.dump((label, transformed_img), f)

        # Original event to keep track of events and images
        with open(data_dir / f"events/{index}.pk", "wb") as f:
            pickle.dump(event, f)


# ## Image creation using multiprocessing

# In[ ]:


import multiprocessing

import tqdm


# In[ ]:


(
    dataset_path,
    full_train_event_list,
    full_train_label_list,
    full_val_event_list,
    full_val_label_list,
) = compute_events_labels(
    peptides=["L1AS5", "L2AS5", "L3AS5", "L5AS5", "L6AS5"],
    validation_percentage=0.2,
    peptide_dir="/data/jhossbach/Peptide_events",
    output_dir="/data/jhossbach/Image_Dataset",
)


# In[ ]:


with multiprocessing.Pool(processes=12) as p:
    # Training dataset
    len_ = len(full_train_event_list)
    p.starmap(
        generate_and_save_img,
        zip(
            full_train_event_list,
            full_train_label_list,
            range(len_),
            len_ * [dataset_path / "train"],
            len_ * [(224, 224)],
            len_ * [False],
            len_ * [False],
        ),
    )

    # Validation dataset
    len_ = len(full_val_event_list)
    p.starmap(
        generate_and_save_img,
        zip(
            full_val_event_list,
            full_val_label_list,
            range(len_),
            len_ * [dataset_path / "validation"],
            len_ * [(224, 224)],
            len_ * [False],
            len_ * [False],
        ),
    )


# In[ ]:




