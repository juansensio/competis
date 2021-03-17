import timm
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .vocab import VOCAB
from Levenshtein import distance
import numpy as np
import torch.nn as nn

# https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/vision_transformer/custom.py

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, P, P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

class Transformer(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.len_vocab = len(VOCAB)
        
        self.patch_embed = PatchEmbedding(self.hparams.img_size, self.hparams.patch_size, 1, self.hparams.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, self.hparams.embed_dim))
        
        self.trg_emb = nn.Embedding(self.len_vocab, self.hparams.embed_dim)
        self.trg_pos_emb = nn.Embedding(self.hparams.max_len, self.hparams.embed_dim)

        dim_feedforward = 4 * self.hparams.embed_dim
        self.transformer = torch.nn.Transformer(
            self.hparams.embed_dim, self.hparams.nhead, self.hparams.num_encoder_layers, self.hparams.num_decoder_layers, dim_feedforward, self.hparams.dropout
        )
        
        self.l = nn.LayerNorm(self.hparams.embed_dim)
        self.fc = nn.Linear(self.hparams.embed_dim, self.len_vocab)

        self.apply(self._init_weights)

    def forward(self, images, captions):
        # embed images
        embed_imgs = self.patch_embed(images)
        embed_imgs = embed_imgs + self.pos_embed  # (B, N, E)
        # embed captions
        #print(captions.shape)
        B, trg_seq_len = captions.shape 
        trg_positions = (torch.arange(0, trg_seq_len).expand(B, trg_seq_len).to(self.device))
        embed_trg = self.trg_emb(captions) + self.trg_pos_emb(trg_positions)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(self.device)
        #tgt_padding_mask = captions == t2ix('PAD')
        # transformer
        y = self.transformer(
            embed_imgs.permute(1,0,2),  # S, B, E
            embed_trg.permute(1,0,2),  # T, B, E
            tgt_mask=trg_mask, # T, T
            #tgt_key_padding_mask = tgt_padding_mask
        ).permute(1,0,2) # B, T, E
        # head
        return self.fc(self.l(y))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def predict(self, images, SOS=1, EOS=2, temp=1):
        self.eval()
        with torch.no_grad():
            images = images.to(self.device)
            B = images.shape[0]
            # start of sentence
            trg_input = torch.tensor([SOS], dtype=torch.long, device=self.device).expand(B, 1)
            while True:
            #for _ in range(self.hparams.max_len):
                logits = self(images, trg_input)[:,-1,:] / temp
                probs = F.softmax(logits, dim=-1) 
                # sample
                pred = torch.multinomial(probs, num_samples=1)
                trg_input = torch.cat([trg_input, pred], 1)
                if torch.any(trg_input == EOS, 1).sum().item() == B or trg_input.shape[1] >= self.hparams.max_len:
                    return trg_input
            #return preds

    def compute_loss(self, batch):
        x, y = batch
        #loss = 0
        #trg_input = y[:,:1] # SOS
        #for i in range(y.shape[1] - 1):
        #    y_hat = self(x, trg_input)
            #loss += F.cross_entropy(y_hat.transpose(1,2), y[:,1:i+2]) 
        #    preds = torch.argmax(y_hat, axis=2)
        #    trg_input = torch.cat([y[:,:1], preds], 1)
        y_hat = self(x, y[:,:-1])
        #loss = F.cross_entropy(y_hat.view(-1, y_hat.size(-1)), y[:,1:].contiguous().view(-1)) 
        loss = F.cross_entropy(y_hat.transpose(1,2), y[:,1:]) 
        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, _ = self.compute_loss(batch)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_loss, y_hat = self.compute_loss(batch)
        #y_hat = torch.argmax(y_hat, axis=2)
        #metric = []
        #for pred, gt in zip(y_hat, y[:,1:]):
            #print(pred, gt)
        #    metric.append(distance(self.decode(pred), self.decode(gt)))
        self.log('val_loss', val_loss, prog_bar=True)
        #self.log('val_ld', np.mean(metric), prog_bar=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(), lr=self.hparams.lr)
        if 'scheduler' in self.hparams:
            schedulers = [
                getattr(torch.optim.lr_scheduler, scheduler)(optimizer, **params)
                for scheduler, params in self.hparams.scheduler.items()
            ]
            return [optimizer], schedulers 
        return optimizer