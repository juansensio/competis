import random


def coin(p=0.5):
    return random.random() < p


def mixup(x, lam):
    x_flipped = x.flip(0).mul_(1 - lam)
    x.mul_(lam).add_(x_flipped)
    return x


def cutmix(x, bb):
    x_flipped = x.flip(0)
    y0, x0, y1, x1 = bb
    x[:, y0:y1, x0:x1] = x_flipped[:, y0:y1, x0:x1]
    return x


def mosaic(x, p):
    x_flipped = x.flip(0)
    py, px = p
    x[:, :py, :px] = x_flipped[:, :py, :px]
    x[:, py:, px:] = x_flipped[:, py:, px:]
    return x
