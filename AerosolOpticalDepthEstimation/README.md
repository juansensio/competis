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
	- tta: 0.92788 / 0.945
- train more (500 eps lr scheduler 5e-3): 0.94319
- data augmentation: 0.941
	- tta:  0.95008 / 0.9653 
- ensamble (5 fold cv): 0.967 
	- best fold (+tta): 0.953 / 0.9658 
- Better heads (MLP)
	- da 512-256-128-1: 0.97323 (0.97420 tta) / 0.97449 
	- cv best fold: 0.97834 (0.99929 tta) / 0.9629 0.96128 (tta)
	- cv ensabmle: 0.97785 (BEST TOP 1)
- fine tune cv mlp: running

fine tune quizás da últimas décimas / centésimas de mejora
mega ensamblado de ensamblados quizás mejora últimas décimas / centésimas

Compare with traditional models (resnet, ...)

## Conclusions

- transfer learning ~ fine tuning (buen modelo preentrenado), unfreezing can give slight boost (but very small)
- data augmentation no mejora, tta solo mejora si entreno con data augmentation (da+tta mejora overall)
- mlp mejora sobre linear head