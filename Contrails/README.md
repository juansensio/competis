baseline r18 t5 e15 -> 0.43907 / 0.427 (test para probar, parece que la metrica se parece en val y test, así que guay)
la baseline tenia un bug en el decoder de la unet, repetir
baseline r18 t5 e100 -> 0.47794
unet r18 t5 da flips e200 -> 0.54259
unet r18 t456 da flips e100 -> running ...

RESULTADOS:

- con da e200 y podría seguir más

PROBAR:

- data augmentation
- loss functions (logcosh, focal)
- loss smooth 1
- optimizer (AdamW)
- lr scheduler
- encoders (resnet34, resnest26d, efficientnet, ...)
- time series
	- pre, during, post (all_bands_t456)
	- all
- threshold optimization
- tta
- ensambling
