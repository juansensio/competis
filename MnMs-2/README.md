challenge -> https://competitions.codalab.org/competitions/31559

baseline todo ceros -> 0

LD 2 channels

unet18 224 bce -> 0.7958 / 0.216
unet18 224 dice -> 0.8085 / 0.218

1 channel / imagen

baseline -> 0.8416 / 0.883
baseline + tta -> 0.8416 / 0.890
baseline + cv + tta -> 0.924 /

probar:

- losses
- más da (crops, deformaciones)
- más resolución (pad if needed)
- más tta
- mejores modelos
- más ensamblado
