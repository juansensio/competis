from src.dm import DataModule
from src.module import Module
import lightning as L
import sys
import yaml
from src.utils import deep_update
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
import torch
import optuna
from optuna.trial import TrialState

config = {
    'encoder': 'resnet34',
    'pretrained': True,
    'optimizer': 'Adam',
    'optimizer_params': {
        'lr': 1e-3
    },
    'scale_factor': 2,
    'loss': 'dice',
    'ckpt_path': None, # resume
    'load_from_checkpoint': None, # load from checkpoint
    'trainer': {
        'accelerator': 'cuda',
        'devices': 1, # con devices 2 el pl me da error al guardar los checkpoints :(
        'max_epochs': 10,
        'logger': True,
        'enable_checkpointing': False,
        'overfit_batches': 0,
        'deterministic': False, # problema con una op del modelo
        'precision': '16-mixed',
        'limit_train_batches': 10,
        'limit_val_batches': 10,
        'log_every_n_steps': 2,
    },
    'datamodule': {
        'batch_size': 64,
        'num_workers': 20,
        'pin_memory': True,
        'bands': list(range(8,17)),
        't': tuple(range(8)),
        'norm_mode': 'mean_std',
        'false_color': True,
        'stats_path': 'stats.csv',
        'train_trans': {}
    },
}


def objective(trial):
    # hparams suggestions
    config['optimizer_params']['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    # config['datamodule']['batch_size'] = trial.suggest_categorical('batch_size', [8, 32, 64, 128])
    # train
    L.seed_everything(42, workers=True)
    dm = DataModule(**config['datamodule'])
    if config['datamodule']['false_color']: config['in_chans'] = 3
    else: config['in_chans'] = len(config['datamodule']['bands'])
    config['t'] = len(config['datamodule']['t'])
    module = Module(config)
    if config['load_from_checkpoint']:
        state_dict = torch.load(config['load_from_checkpoint'])['state_dict']
        module.load_state_dict(state_dict)
    config['trainer']['callbacks'] = [EarlyStopping(monitor="val_metric", min_delta=0.00, patience=3, verbose=False, mode="max")] 
    wandb_logger = None
    if config['trainer']['logger']:
        wandb_logger = WandbLogger(
            project="Contrails-hpopt",
            name=f'lr={config["optimizer_params"]["lr"]:.5f}-bs={config["datamodule"]["batch_size"]}',
            config=config
        )
        config['trainer']['logger'] = wandb_logger
        if 'scheduler' in config and config['scheduler']:
            config['trainer']['callbacks'] += [
                LearningRateMonitor(logging_interval='step')]
    trainer = L.Trainer(**config['trainer'])
    torch.set_float32_matmul_precision('medium')
    # module = torch.compile(module)
    trainer.fit(module, dm, ckpt_path=config['ckpt_path'])
    if wandb_logger: wandb_logger.experiment.finish()
    return trainer.callback_metrics['val_metric'].item()


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
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
