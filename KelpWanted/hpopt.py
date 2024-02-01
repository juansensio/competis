from src import DataModule, Module
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
import torch
import optuna
from optuna.trial import TrialState

config = {
    "encoder": "timm-resnest50d",
    "pretrained": True,
    "optimizer": "Adam",
    "optimizer_params": {
        "lr": 1e-3,
    },
    "loss": "dice",
    "in_chans": 5,
    "mask_loss": True,
    "architecture": "UnetPlusPlus",
    "padding": 1,
    "ckpt_path": None,  # resume
    "load_from_checkpoint": None,  # load from checkpoint
    "trainer": {
        "accelerator": "cuda",
        "devices": 1,  # con devices 2 el pl me da error al guardar los checkpoints :(
        "max_epochs": 15,
        "logger": None,
        "enable_checkpointing": False,
        "overfit_batches": 0,
        "precision": "16-mixed",
        "deterministic": True,
    },
    "datamodule": {
        "batch_size": 32,
        "num_workers": 20,
        "pin_memory": True,
        "train_trans": {},
        "Dataset": "DatasetFCIm",
    },
}


def objective(trial):
    # hparams suggestions
    config["optimizer_params"]["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    config["seed"] = trial.suggest_int("seed", 0, 1000)
    # config['datamodule']['batch_size'] = trial.suggest_categorical('batch_size', [8, 32, 64, 128])
    # train
    L.seed_everything(config["seed"], workers=True)
    dm = DataModule(**config["datamodule"])
    module = Module(config)
    config["trainer"]["callbacks"] = [
        EarlyStopping(
            monitor="val_metric", min_delta=0.00, patience=5, verbose=False, mode="max"
        )
    ]
    wandb_logger = WandbLogger(
        project="Kelp-hpopt",
        name=f'lr={config["optimizer_params"]["lr"]:.5f}-seed={config["seed"]}',
        config=config,
    )
    config["trainer"]["logger"] = wandb_logger
    trainer = L.Trainer(**config["trainer"])
    torch.set_float32_matmul_precision("medium")
    trainer.fit(module, dm, ckpt_path=config["ckpt_path"])
    wandb_logger.experiment.finish()
    return trainer.callback_metrics["val_metric"].item()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
