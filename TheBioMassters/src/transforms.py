import torch 

class BaseTransform(torch.nn.Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p 

    def forward(self, x, y):
        raise NotImplementedError

class RandomHorizontalFlip(BaseTransform):
    def forward(self, x, y):
        if torch.rand(1) < self.p:
            x = torch.flip(x, [4])
            y = torch.flip(y, [2])
            return x, y
        return x, y

class RandomVerticalFlip(BaseTransform):
    def forward(self, x, y):
        if torch.rand(1) < self.p:
            x = torch.flip(x, [3])
            y = torch.flip(y, [1])
            return x, y
        return x, y

class RandomTranspose(BaseTransform):
    def forward(self, x, y):
        if torch.rand(1) < self.p:
            x = torch.transpose(x, 3, 4)
            y = torch.transpose(y, 1, 2)
            return x, y
        return x, y

class RandomRotate90(BaseTransform):
    def forward(self, x, y):
        if torch.rand(1) < self.p:
            r = torch.randint(1,4,(1,)).item()
            x = torch.rot90(x, r, (3,4))
            y = torch.rot90(y, r, (1,2))
            return x, y
        return x, y