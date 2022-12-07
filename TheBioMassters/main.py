from src.dali import Dataloader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from fastprogress.fastprogress import master_bar, progress_bar
from src.models.unet_df2 import UNetDF2
import numpy as np
import wandb

# print(torch.__version__, torch.cuda.is_available())

BATCH_SIZE = 16
EPOCHS = 500
USE_AMP = True
VAL_SIZE = 0.2
SEED = 42
CHECKPOINTING = True
LOG_FREQ = 30
NAME = 'dfti_opt'
LOAD = False

train = pd.read_csv('data/train_chip_ids.csv')
train, val = train_test_split(train, test_size=VAL_SIZE, random_state=SEED)
dl = {
    'train': Dataloader(train.chip_id.values, BATCH_SIZE, trans=True, seed=SEED, shuffle=True),
    'val': Dataloader(val.chip_id.values, BATCH_SIZE)
}
model = UNetDF2()
if LOAD:
    model.load_state_dict(torch.load(LOAD))
model.cuda()
# model = torch.compile(model, backend="inductor").cuda() # peta pero el ejemplo kk funciona, probablemente haya que toquetear el modelo
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
if LOG_FREQ > 0:
    wandb.init(project="TheBioMassters-yt", name=NAME)
    wandb.watch(model, log_freq=LOG_FREQ)
mb = master_bar(range(1, EPOCHS+1))
best_metric = 1e10
for epoch in mb:
    model.train()
    losses, metric = [], []
    for batch_idx, data in progress_bar(enumerate(dl['train']), total=len(train)//BATCH_SIZE, parent=mb):
        batch_data = data[0]
        x1, x2, y = batch_data['x1'], batch_data['x2'], batch_data['labels'].squeeze(
            -1)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            y_hat = model(x1, x2)
            loss = torch.mean(torch.sqrt(
                torch.mean((y_hat - y)**2, dim=(1, 2))))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        metrics = torch.mean(torch.sqrt(torch.mean(
            (y_hat * 12905.3 - y * 12905.3)**2, dim=(1, 2))))
        losses.append(loss.item())
        metric.append(metrics.item())
        if LOG_FREQ > 0 and batch_idx % LOG_FREQ == 0:
            wandb.log({'loss': loss, 'metric': metrics, 'epoch': epoch})
        mb.child.comment = f'loss: {loss.item():.4f}, metric: {metrics.item():.4f}'
    model.eval()
    val_loss, val_metric = [], []
    for data in progress_bar(dl['val'], total=len(val)//BATCH_SIZE, parent=mb):
        batch_data = data[0]
        x1, x2, y = batch_data['x1'], batch_data['x2'], batch_data['labels'].squeeze(
            -1)
        with torch.no_grad():
            y_hat = model(x1, x2)
            loss = torch.mean(torch.sqrt(
                torch.mean((y_hat - y)**2, dim=(1, 2))))
            metrics = torch.mean(torch.sqrt(torch.mean(
                (y_hat * 12905.3 - y * 12905.3)**2, dim=(1, 2))))
        val_loss.append(loss.item())
        val_metric.append(metrics.item())
        mb.child.comment = f'val_loss: {np.mean(val_loss):.4f}, val_metric: {np.mean(val_metric):.4f}'
    val_metric = np.mean(val_metric)
    if LOG_FREQ > 0:
        wandb.log({'val_loss': np.mean(val_loss),
                  'val_metric': val_metric, 'epoch': epoch})
    if CHECKPOINTING:
        torch.save(model.state_dict(),
                   f'checkpoints/{NAME}-epoch={epoch}.ckpt')
        if val_metric < best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(
            ), f'checkpoints/{NAME}-val_metric={val_metric:.4f}-epoch={epoch}.ckpt')
    mb.write(f'Epoch {epoch}/{EPOCHS} loss: {np.mean(losses):.4f} metric: {np.mean(metric):.4f} val_loss: {np.mean(val_loss):.4f} val_metric: {val_metric:.4f}')
