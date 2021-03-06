import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.dm import DataModule
from src.utils import parse_config
from src.models import SMP
import optuna
from src.cbs import PyTorchLightningPruningCallback

def objective(trial):
    pl.seed_everything(42, workers=True)
    trans = {'Resize': {'width': 224, 'height' :224}}
    dm = DataModule(
        batch_size=64,
        num_workers=24,
        pin_memory=True,
        train_trans=trans,
        val_trans=trans,
        shuffle_train=False
    )
    loss = trial.suggest_categorical("loss", ["bce", "dice", "jaccard", "focal", "log_cosh_dice"])
    model = SMP({
        'optimizer': 'Adam',
        'lr': 0.0003,
        'loss': loss,
        'model': 'Unet', 
        'backbone': 'resnet18',
        'pretrained': 'imagenet'
    })
    wandb_logger = WandbLogger(project="MnMs2-opt", name=loss)
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        logger=wandb_logger,
        max_epochs=10,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_iou")],
        checkpoint_callback=False,
        limit_train_batches=1.,
        limit_val_batches=1.,
        deterministic=True
    )
    trainer.fit(model, dm)
    score = trainer.test(model, dm.val_dataloader())
    wandb_logger.experiment.finish()
    return score[0]['test_iou']

#sampler = optuna.samplers.TPESampler(seed=42)
#study = optuna.create_study(direction='maximize', sampler=sampler)
#study.optimize(objective, n_trials=100)

search_space = {"loss": ["bce", "dice", "jaccard", "focal", "log_cosh_dice"]}
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, n_trials=5)
print(study.best_params)
