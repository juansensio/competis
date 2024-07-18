# Aerosol Optical Depth Estimation

https://solafune.com/competitions/ca6ee401-eba9-4f7d-95e6-d1b378a17200?menu=about&tab=overview

Aeronet dataset: https://aeronet.gsfc.nasa.gov/

## Results

Trying clay model 

- basline (bs 32 30 eps lr 1e-4): 0.76286 / 0.8174
- baseline (bs 64 100 eps lr 3e-4): 0.83894 / 0.88166
- fine tune encoder (bs 64 100 eps lr 3e-4): 0.84048 
- data augmentation: 0.82154
- train more (200 eps lr scheduler 1e-3): 0.92938 / 0.9505
- train more (500 eps lr scheduler 1e-3): 0.941 / 0.9588
	- tta: 0.9088 / 0.945
- train more (500 eps lr scheduler 5e-3): 0.94319
- data augmentation: running ...
- ensamble (5 fold cv)
- tta

Compare with traditional models (resnet, ...)

## Conclusions

- transfer learning ~ fine tuning (buen modelo preentrenado)
- baseline bs 63 100 eps lr 1e-4 sigue aprendiendo (más epochs, o subir lr)