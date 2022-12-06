from src.dali import Dataloader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from fastprogress.fastprogress import master_bar, progress_bar
from src.models.unet_df2 import UNetDF2
import numpy as np

print(torch.__version__, torch.cuda.is_available())

BATCH_SIZE = 8
EPOCHS = 10
VAL_SIZE = 0.2
SEED = 42

train = pd.read_csv('data/train_chip_ids.csv')
train, val = train_test_split(
    train[:1000], test_size=VAL_SIZE, random_state=SEED)
dl = {
    'train': Dataloader(train.chip_id.values, BATCH_SIZE),
    'val': Dataloader(val.chip_id.values, BATCH_SIZE)
}
model = UNetDF2().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
mb = master_bar(range(1, EPOCHS+1))
for epoch in mb:
    model.train()
    losses, metric = [], []
    for data in progress_bar(dl['train'], total=len(train)//BATCH_SIZE, parent=mb):
        batch_data = data[0]
        x1, x2, y = batch_data['x1'], batch_data['x2'], batch_data['labels']
        optimizer.zero_grad()
        y_hat = model(x1, x2)
        loss = torch.mean(torch.sqrt(torch.mean((y_hat - y)**2, dim=(1, 2))))
        loss.backward()
        optimizer.step()
        metrics = torch.mean(torch.sqrt(torch.mean(
            (y_hat * 12905.3 - y * 12905.3)**2, dim=(1, 2))))
        losses.append(loss.item())
        metric.append(metrics.item())
        mb.child.comment = f'loss: {np.mean(losses):.4f}, metric: {np.mean(metric):.4f}'
    model.eval()
    val_loss, val_metric = [], []
    for data in progress_bar(dl['val'], total=len(val)//BATCH_SIZE, parent=mb):
        batch_data = data[0]
        x1, x2, y = batch_data['x1'], batch_data['x2'], batch_data['labels']
        with torch.no_grad():
            y_hat = model(x1, x2)
            loss = torch.mean(torch.sqrt(
                torch.mean((y_hat - y)**2, dim=(1, 2))))
            metrics = torch.mean(torch.sqrt(torch.mean(
                (y_hat * 12905.3 - y * 12905.3)**2, dim=(1, 2))))
        val_loss.append(loss.item())
        val_metric.append(metrics.item())
        mb.child.comment = f'val_loss: {np.mean(val_loss):.4f}, val_metric: {np.mean(val_metric):.4f}'
    mb.write(f'Epoch {epoch}/{EPOCHS} loss: {np.mean(losses):.4f} metric: {np.mean(metric):.4f} val_loss: {np.mean(val_loss):.4f} val_metric: {np.mean(val_metric):.4f}')
