challenge -> https://competitions.codalab.org/competitions/31559

baseline todo ceros -> 0

LD 2 channels

unet18 224 bce -> 0.7958 / 0.216
unet18 224 dice -> 0.8085 / 0.218

1 channel / imagen

baseline bce 150e da flips -> 0.8416 / 0.883
baseline bce 150e da flips + tta -> 0.8416 / 0.890
baseline bce 150e da flips + cv + tta -> 0.924 / 0.891

seed everything deterministic 50 da flips

baseline bce -> 0.8339 (of)
baseline focal -> 0.8279 / 0.837 (of) (focal peor)
baseline bce crops -> 0.8214
baseline bce crops distortion -> 0.8220
baseline bce crops distortion blur -> 0.8191
baseline bce no flips -> 0.8275 (of)
baseline bce distortion -> 0.8293 (of)

models

deeplabv3++ se_resnext50 lrs3 da 224 bs64 70e -> 0.8321 (podía seguir mejorando)
unet enb3 lrs5 da 224 bs32 100e -> 0.853 / 0.872 (mejora val, peor test)
unet enb3 lrs5 da 224 bs64 100e dice -> 0.8166 / 0.803 (dice da peor)

bce 200e flips + crops + dist

unet resnet18 -> 0.8476 / 0.884
depplabv3++ resnet18 -> harley ...

lr5

unet resnet18 -> 0.8414
