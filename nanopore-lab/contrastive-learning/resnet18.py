# Custom imports
from resnet import *
from data import *
from lightning_modules import *
from losses import *

# Lightning imports
from lightning.pytorch.callbacks import StochasticWeightAveraging, GradientAccumulationScheduler
import lightning.pytorch as pl

# Torch imports
from torchvision.transforms import v2
from torch import optim

# System imports
from dataclasses import dataclass
import glob


@dataclass
class Parameters:
    storage_path: str
    experiment_name: str
    image_size: int
    epochs: int
    learning_rate: float
    batch_size: int
    batch_accumulation: int


model = resnet18(num_classes=42, in_channels=2, scale_factor=1)
parameters = Parameters(
    storage_path="/work/stovey/contrastive/resnet18",
    experiment_name="resnet18",
    image_size=244,
    epochs=10000,
    learning_rate=1e-4,
    batch_size=100,
    batch_accumulation=100
)

# Define logger (insert favorite logger)
logger = pl.loggers.MLFlowLogger(
    experiment_name=parameters.experiment_name, 
    save_dir=parameters.storage_path
)

# Checkpointing for training state.
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="ptl/validation_loss",
    dirpath=parameters.storage_path + "/ckpts",
    filename="resnet18-{epoch:02d}-{ptl/validation_loss:.2f}",
    save_top_k=2,
    mode="min",
)

grad_accumulator = GradientAccumulationScheduler(scheduling={0: 500, 10: 100, 20: 50, 50: 10})
swa = StochasticWeightAveraging(swa_lrs=1e-2)
# Lightning trainer definition
trainer = pl.Trainer(
    accelerator="gpu",
    num_nodes=2,
    logger=logger,
    # accumulate_grad_batches=parameters.batch_accumulation,
    callbacks=[swa, grad_accumulator, checkpoint_callback],
    devices=-1,
    strategy="ddp",
    max_epochs=parameters.epochs,
    log_every_n_steps=10,
    default_root_dir=parameters.storage_path,
    enable_progress_bar=True,
    sync_batchnorm=True,
)

# Define the normalization transform
mean = torch.tensor([3.6632e-05, 6.1436e-06])
std = torch.tensor([0.0169, 0.0168])
train_transform = v2.Compose(
    [
        # v2.RandomErasing(0.1),
        # v2.RandomHorizontalFlip(0.2),
        # v2.RandomVerticalFlip(0.2),
        v2.Normalize(mean, std),
    ]
)
test_transform = v2.Compose(
    [
        v2.Normalize(mean, std),
    ]
)

# Use the custom datamodule
datamodule = DataModule(
    data_dir="/work/jhossbach/Image_Dataset_old/dataset_all",
    batch_size=parameters.batch_size,
    train_transform=train_transform,
    test_transform=test_transform
)

# Optimizer and scheduler
# optimizer = optim.Adam(model.parameters(), lr=parameters.learning_rate, weight_decay=1e-5)
optimizer = optim.SGD(model.parameters(), lr=parameters.learning_rate, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
loss_fn = CombinedLoss(margin=1.0, contrastive_weight=0.2, supervised_weight=1.0)

# Lightning model definition
lit_model = LitResModel(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_module=loss_fn
)

# ckpts = glob.glob(f"/work/stovey/contrastive/resnet18/ckpts/*")
# epochs = [int(ckpt.split("/")[-1].split("-")[1].split("=")[-1]) for ckpt in ckpts]
# max_epoch_location = epochs.index(max(epochs))
# ckpt_path = glob.glob(f"{ckpts[max_epoch_location]}/*")[0]

# # Start training
trainer.fit(lit_model, datamodule=datamodule, ckpt_path=None)