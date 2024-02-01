from src import DataModule, Module
import lightning as L
import sys
import yaml
from src.utils import deep_update
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
import torch

config = {
    "encoder": "resnet18",
    "pretrained": True,
    "optimizer": "Adam",
    "optimizer_params": {
        "lr": 1e-3,
    },
    "loss": "dice",
    "in_chans": 3,
    "padding": 1,
    "ckpt_path": None,  # resume
    "load_from_checkpoint": None,  # load from checkpoint
    "trainer": {
        "accelerator": "cuda",
        "devices": 1,  # con devices 2 el pl me da error al guardar los checkpoints :(
        "max_epochs": 30,
        "logger": None,
        "enable_checkpointing": False,
        "overfit_batches": 0,
        "precision": "16-mixed",
    },
    "datamodule": {
        "batch_size": 64,
        "num_workers": 20,
        "pin_memory": True,
        "train_trans": {},
    },
}


def train(config, name):
    # L.seed_everything(42, workers=True)
    dm = DataModule(**config["datamodule"])
    module = Module(config)
    if config["load_from_checkpoint"]:
        state_dict = torch.load(config["load_from_checkpoint"])["state_dict"]
        module.load_state_dict(state_dict)
    config["trainer"]["callbacks"] = []
    if config["trainer"]["enable_checkpointing"]:
        config["trainer"]["callbacks"] += [
            ModelCheckpoint(
                dirpath="./checkpoints",
                filename=f"{name}-{{val_metric:.5f}}-{{epoch}}",
                monitor="val_metric",
                mode="max",
            ),
            ModelCheckpoint(  # save last epoch
                dirpath="./checkpoints",
                filename=f"{name}-{{epoch}}",
            ),
            # EarlyStopping(monitor="val_metric", patience=5, mode="max", verbose=True),
        ]
    if config["trainer"]["logger"]:
        config["trainer"]["logger"] = WandbLogger(
            project="Kelp", name=name, config=config
        )
        if "scheduler" in config and config["scheduler"]:
            config["trainer"]["callbacks"] += [
                LearningRateMonitor(logging_interval="step")
            ]
    trainer = L.Trainer(**config["trainer"])
    torch.set_float32_matmul_precision("medium")
    # module = torch.compile(module)
    trainer.fit(module, dm, ckpt_path=config["ckpt_path"])


if __name__ == "__main__":
    name = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        name = config_file[:-4]
        if config_file:
            with open(config_file, "r") as stream:
                loaded_config = yaml.safe_load(stream)
            deep_update(config, loaded_config)
    print(config)
    train(config, name)
