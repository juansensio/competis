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
        self.save_hyperparameters(config)
        self.encoder = Encoder(self.hparams.backbone, self.hparams.pretrained)
        self.decoder = Decoder()

    # def forward(self, x):
    #     z = self.encoder(x)
    #     bs, _ = z.size()
    #     i = torch.ones((bs, 1), dtype=torch.long, device=self.device)*t2ix('SOS')
    #     h = z.unsqueeze(0) # works only for 1 rnn layer
    #     preds = torch.tensor([], dtype=torch.long, device=self.device)
    #     while True:
    #         o, h = self.decoder(i, h)
    #         i = torch.argmax(o, axis=1).view(bs, 1)
    #         preds = torch.cat([preds, i], axis=1)
    #         if torch.any(preds == t2ix('EOS'), 1).sum().item() == bs or preds.shape[1] >= self.hparams.max_len:
    #             break
    #     return preds

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.decoder(z)
        return y_hat

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            preds = self(x)
            preds = torch.argmax(preds, axis=2)
            decoded_batch = []
            for pred in preds:
                decoded = decode(pred).split('EOS')[0]
                #decoded = decoded.split('PAD')[0]
                decoded_batch.append(decoded)
            return decoded_batch

    # def compute_loss(self, batch):
    #     x, y = batch
    #     z = self.encoder(x)
    #     bs, _ = z.size()
    #     i = torch.ones((bs, 1), dtype=torch.long, device=self.device)*t2ix('SOS')
    #     h = z.unsqueeze(0) # works only for 1 rnn layer
    #     loss = 0
    #     preds = torch.tensor([], dtype=torch.long, device=self.device)
    #     for k in range(y.shape[1]):
    #         o, h = self.decoder(i, h)
    #         loss += F.cross_entropy(o, y[:, k].view(bs)) 
    #         # usamos predicción como siguiente input    
    #         i = torch.argmax(o, axis=1).view(bs, 1)
    #         # usamos gt como siguiente input 
    #         #i = y[:, i].view(bs, 1)
    #         preds = torch.cat([preds, i], axis=1)
    #     return loss, preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #print(y_hat.shape)
        # [ Batch, seq len, num features ] --> [ Batch, num features, seq len ]
        loss = F.cross_entropy(y_hat.permute(0,2,1), y)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # [ Batch, seq len, num features ] --> [ Batch, num features, seq len ]
        val_loss = F.cross_entropy(y_hat.permute(0,2,1), y)
        y_hat = torch.argmax(y_hat, axis=2)
        metric = []
        for pred, gt in zip(y_hat, y):
            metric.append(distance(decode(pred).split('EOS')[0], decode(gt).split('EOS')[0]))
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
        # features = self.backbone(x)   
        # bs, c, h, w = features[-1].size()
        # return features[-1].view(bs, c, -1)

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
        
# class Decoder(torch.nn.Module):
#     def __init__(self, len_vocab=len(VOCAB), d_model=16, nhead=2, num_layers=4):
#         super().__init__()
#         self.len_vocab = len_vocab
#         #self.embedding = torch.nn.Embedding(len(VOCAB), 100)
#         encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model)
#         self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.head = torch.nn.Linear(d_model, len_vocab)
        
#     def forward(self, x):
#         # batch, len 
#         #e = self.embedding(x)
#         # batch, len, features
#         o = self.transformer(x)
#         # batch x len, features
#         o = self.head(o.contiguous().view(-1, o.size(-1)))
#         return o.contiguous().view(x.size(0), -1, o.size(-1))

