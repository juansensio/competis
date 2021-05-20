challenge -> https://competitions.codalab.org/competitions/31559

baseline todo ceros -> 0

LD 2 channels

unet18 224 bce -> 0.7958 / 0.216
unet18 224 dice -> 0.8085 / 0.218

1 channel / imagen

baseline bce -> 0.8416 / 0.883
baseline bce + tta -> 0.8416 / 0.890
baseline bce + cv + tta -> 0.924 / 0.891

probar:

- losses: en 10 epochs exp -> focal > jaccard > bce > dice > log_cosh_dice
- más da (crops, deformaciones)
- más resolución (pad if needed)
- más tta
- mejores modelos
- lr scheduling
- más ensamblado
