from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class MyEarlyStopping(EarlyStopping):

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < pl_module.hparams.es_start_from:
            print(f"\nSkipping early stopping until epoch {pl_module.hparams.es_start_from}\n")
            pass
        else:
            super().on_validation_end(trainer, pl_module)