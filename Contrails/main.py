from src.dm import DataModule
from src.module import Module
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
    "freeze": False,
    "optimizer": "Adam",
    "optimizer_params": {
        "lr": 1e-3,
    },
    "loss": "dice",
    "ckpt_path": None,  # resume
    "load_from_checkpoint": None,  # load from checkpoint
    "trainer": {
        "accelerator": "cuda",
        "devices": 1,  # con devices 2 el pl me da error al guardar los checkpoints :(
        "max_epochs": 30,
        "logger": None,
        "enable_checkpointing": False,
        "overfit_batches": 0,
        "deterministic": False,  # problema con una op del modelo
        "precision": "16-mixed",
    },
    "datamodule": {
        "batch_size": 64,
        "num_workers": 10,
        "pin_memory": True,
        "bands": list(range(8, 17)),
        "t": [4],  # tuple(range(8)),
        "norm_mode": "mean_std",
        "false_color": False,
        "bn": False,
        "stats_path": "/fastdata/contrails/stats.csv",
        "train_trans": {  # NO PONER NADA QUE HAGA RESIZE AQUI !!!
            # 'HorizontalFlip': {'p': 0.5},
            # 'VerticalFlip': {'p': 0.5},
            # 'RandomRotate90': {'p': 0.5},
            # 'Transpose': {'p': 0.5}
        },
        "input_size": (256, 256),  # PONERLO AQUI, SOLO AFECTA A INPUTS
    },
}


def train(config, name):
    L.seed_everything(42, workers=True)
    dm = DataModule(**config["datamodule"])
    if config["datamodule"]["false_color"]:
        config["in_chans"] = 3
    elif config["datamodule"]["bn"]:
        config["in_chans"] = 1
    else:
        config["in_chans"] = len(config["datamodule"]["bands"])
    config["t"] = len(config["datamodule"]["t"])
    module = Module(config)
    if config["load_from_checkpoint"]:
        state_dict = torch.load(config["load_from_checkpoint"])["state_dict"]
        module.load_state_dict(state_dict)
    config["trainer"]["callbacks"] = []
    if config["trainer"]["enable_checkpointing"]:
        # esto falla en multi-gpu
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
            project="Contrails", name=name, config=config
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
