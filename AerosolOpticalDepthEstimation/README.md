# Aerosol Optical Depth Estimation

https://solafune.com/competitions/ca6ee401-eba9-4f7d-95e6-d1b378a17200?menu=about&tab=overview

Aeronet dataset: https://aeronet.gsfc.nasa.gov/

## Results

Trying clay model 

- basline (bs 32, 30 eps): 0.76286 / 0.8174
- baseline (bs 64, 100 eps): 0.83894 / 0.88166
- fine tune encoder
- data augmentation
- lr scheduler
- ensamble (5 fold cv)

Compare with traditional models (resnet, ...)

## Conclusions

- transfer learning ~ fine tuning (buen modelo preentrenado)
- baseline bs 63 100 eps lr 1e-4 sigue aprendiendo (más epochs, o subir lr)