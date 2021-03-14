import timm
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .vocab import VOCAB, decode, t2ix
from Levenshtein import distance
import numpy as np

class Baseline(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        config = self.check_config(config)
        self.save_hyperparameters(config)
        self.encoder = Encoder(self.hparams.backbone, self.hparams.pretrained)
        self.decoder = Decoder()

    def check_config(self, config):
        if config is None:
            config = {} 
        # default config
        config['max_len'] = 20 if not 'max_len' in config else config['max_len']
        return config

    def forward(self, x):
        z = self.encoder(x)
        bs, _ = z.size()
        i = torch.ones((bs, 1), dtype=torch.long, device=self.device)*t2ix('SOS')
        h = z.unsqueeze(0) # works only for 1 rnn layer
        preds = torch.tensor([], dtype=torch.long, device=self.device)
        while True:
            o, h = self.decoder(i, h)
            i = torch.argmax(o, axis=1).view(bs, 1)
            preds = torch.cat([preds, i], axis=1)
            if torch.any(preds == t2ix('EOS'), 1).sum().item() == bs or preds.shape[1] >= self.hparams.max_len:
                break
        return preds

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            preds = self(x)
            decoded_batch = []
            for pred in preds:
                decoded = decode(pred).split('EOS')[0]
                #decoded = decoded.split('PAD')[0]
                decoded_batch.append(decoded)
            return decoded_batch

    def compute_loss(self, batch):
        x, y = batch
        z = self.encoder(x)
        bs, _ = z.size()
        i = torch.ones((bs, 1), dtype=torch.long, device=self.device)*t2ix('SOS')
        h = z.unsqueeze(0) # works only for 1 rnn layer
        loss = 0
        preds = torch.tensor([], dtype=torch.long, device=self.device)
        for k in range(y.shape[1]):
            o, h = self.decoder(i, h)
            loss += F.cross_entropy(o, y[:, k].view(bs)) 
            # usamos predicción como siguiente input    
            i = torch.argmax(o, axis=1).view(bs, 1)
            # usamos gt como siguiente input 
            #i = y[:, i].view(bs, 1)
            preds = torch.cat([preds, i], axis=1)
        return loss, preds

    def training_step(self, batch, batch_idx):
        loss, y_hat = self.compute_loss(batch)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_loss, y_hat = self.compute_loss(batch)
        metric = []
        for pred, gt in zip(y_hat, y):
            metric.append(distance(decode(pred), decode(gt)))
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_ld', np.mean(metric), prog_bar=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(), lr=self.hparams.lr)
        if 'scheduler' in self.hparams:
            schedulers = [
                getattr(torch.optim.lr_scheduler, scheduler)(optimizer, **params)
                for scheduler, params in self.hparams.scheduler.items()
            ]
            return [optimizer], schedulers 
        return optimizer

class Encoder(torch.nn.Module):
    def __init__(self, backbone='resnet18', pretrained=False, in_channels=1):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=(1,1)),
            torch.nn.Flatten()
        )
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features[-1])

class Decoder(torch.nn.Module):
    def __init__(self, len_vocab=len(VOCAB), hidden_size=512, num_layers=1):
        super().__init__()
        #self.embedding = torch.nn.Embedding(len(VOCAB), 100)
        self.len_vocab = len_vocab
        self.rnn = torch.nn.GRU(self.len_vocab, hidden_size, num_layers=num_layers, batch_first=True)
        self.out = torch.nn.Linear(hidden_size, self.len_vocab)
        
    def forward(self, x, h):
        # batch x len
        #e = self.embedding(x)
        e = torch.nn.functional.one_hot(x, num_classes=self.len_vocab).float()
        # batch x len x features
        o, h = self.rnn(e, h)
        o = self.out(o.squeeze(1))
        return o, h