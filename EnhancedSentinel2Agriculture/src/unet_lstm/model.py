import torch.nn.functional as F 
import torch 
import pytorch_lightning as pl
from torchmetrics import MatthewsCorrcoef
from ..convlstm.ConvLSTM import ConvLSTMCell
import torch.nn as nn

class EncoderConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, mp = True):
        super().__init__()

        self.conv1 = ConvLSTMCell(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            kernel_size=kernel_size, 
            bias=bias
        )
        #self.bn1 = torch.nn.BatchNorm2d(hidden_dim)

        self.conv2 = ConvLSTMCell(
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim, 
            kernel_size=kernel_size, 
            bias=bias
        )
        #self.bn2 = torch.nn.BatchNorm2d(hidden_dim)

        self.mp = mp
        self.maxpool = torch.nn.MaxPool2d(2)

    def forward(self, input_tensor, hidden_state=None):
        
        seq_len = input_tensor.size(1)

        if self.mp:
            x_mp = []
            for t in range(seq_len):
                x = input_tensor[:, t, :, :, :]
                x = self.maxpool(x)
                x_mp.append(x)
            input_tensor = torch.stack(x_mp, dim=1)

        
        b, _, _, h, w = input_tensor.size()
        
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        cur_layer_input = input_tensor

        h, c = hidden_state[0]
        output_inner = []
        for t in range(seq_len):
            x = cur_layer_input[:, t, :, :, :]
            h, c = self.conv1(x, cur_state=[h, c])
            #h = self.bn1(h)
            output_inner.append(h)
        
        layer_output = torch.stack(output_inner, dim=1)
        cur_layer_input = layer_output

        h, c = hidden_state[1]
        output_inner = []
        for t in range(seq_len):
            x = cur_layer_input[:, t, :, :, :]
            h, c = self.conv2(x, cur_state=[h, c])
            #h = self.bn2(h)
            output_inner.append(h)
                    
        layer_output = torch.stack(output_inner, dim=1)

        return layer_output

    def _init_hidden(self, batch_size, image_size):
        return [
            self.conv1.init_hidden(batch_size, image_size),
            self.conv2.init_hidden(batch_size, image_size),
        ]

def conv3x3_bn(ci, co):
    return torch.nn.Sequential(
        torch.nn.Conv2d(ci, co, 3, padding=1),
        torch.nn.BatchNorm2d(co),
        torch.nn.ReLU(inplace=True)
    )

class DecoderConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.upsample = torch.nn.ConvTranspose2d(ci, co, 2, stride=2)
        self.conv1 = conv3x3_bn(ci, co)
        self.conv2 = conv3x3_bn(co, co)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX, 0, diffY, 0))
        # concatenamos los tensores
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UnetLSTM(nn.Module):

    def __init__(self, input_dim = 3, hidden_dim = [16], kernel_size = 3, bias=True, num_classes = 1):
        super(UnetLSTM, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        num_layers = len(hidden_dim)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias

        encoder_list = [
            EncoderConv(
                self.input_dim,
                self.hidden_dim[0],
                self.kernel_size[0],
                self.bias,
                mp=False
            )
        ]
        for i in range(1, self.num_layers):
            encoder_list.append(
                EncoderConv(
                    self.hidden_dim[i-1],
                    self.hidden_dim[i],
                    self.kernel_size[i],
                    self.bias,
                    mp=True
                )
            )        

        self.encoder = nn.ModuleList(encoder_list)

        decoder_list = []
        for i in reversed(range(1, self.num_layers)):
            decoder_list.append(
                DecoderConv(
                    self.hidden_dim[i],
                    self.hidden_dim[i-1],
                )
            )      

        self.decoder = nn.ModuleList(decoder_list)

        self.out = torch.nn.ConvTranspose2d(self.hidden_dim[0], num_classes, 4, stride=4)


    def forward(self, input_tensor, hidden_state=None):
        features = []
        h = input_tensor
        for e in self.encoder:
            h = e(h, hidden_state)
            output = h[:,-1,...]
            features.append(output)
        x = features[-1]
        for i, d in enumerate(self.decoder):
            encoder_output = features[self.num_layers - i - 2]
            x = d(x, encoder_output)
        return self.out(x)


    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class UnetLSTMModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        self.model = UnetLSTM(
            input_dim=3,
            hidden_dim=self.hparams.hidden_dim,
            kernel_size=(3, 3),
            bias=True,
        )
        
        self.metric = MatthewsCorrcoef(num_classes=2)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            y_hat = self(x)
            return torch.sigmoid(y_hat)

    def shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        return y_hat, loss
        
    def training_step(self, batch, batch_idx):
        _, loss = self.shared_step(batch)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, y = batch
        y_hat, loss = self.shared_step(batch)
        metric = self.metric(torch.sigmoid(y_hat), y.long())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mathCorrCoef', metric, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)