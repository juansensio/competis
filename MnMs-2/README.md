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

baseline bce -> 0.8339
baseline focal -> 0.8279 / 0.837
baseline bce crops -> running ...
baseline bce crops distortion ->
baseline bce crops distortion blur ->

probar:

- losses: bce > focal > dice
- más da (crops, deformaciones)
- más resolución (384, pad if needed)
- lr scheduling
- más tta
- mejores modelos
- más ensamblado
