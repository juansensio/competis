baseline r18 t5 e15 -> 0.43907 / 0.427 (test para probar, parece que la metrica se parece en val y test, así que guay)
la baseline tenia un bug en el decoder de la unet, repetir
baseline r18 t5 e100 -> 0.47794
unet r18 t5 da flips e200 -> 0.54259
unet r18 t456 da flips e200 -> 0.57122
REFACTOR
unet r18 fc t5 da flips e100 -> 0.567
unet r18 all_bands t5 da flips e100 min_max -> mal
unet r18 all_bands t5 da flips e100 mean_std -> 0.54979
unet r18 fc t456 da flips e100 -> 0.58363
unet r18 fc t456 da flips e200 lrsch -> 0.58386 
unet r18 fc t456 da flips e200 lrsch AdamW -> 0.57889
unet resnet34 fc t456 da flips e100  -> 0.59515 / 0.604 / 0.605 (BEST, podría seguir aprendiendo)
unet resnest26d fc t456 da flips e100 lr3e-4 (peta con lr1e-3) -> 0.60265 (ha petado, usar lr scheduler)
unet resnet34 fc all t da flips e100 -> running ...

RESULTADOS:

- añadir da (flips) mejora
- false color > all bands(mean_std) > all_bands (min_max) (metric & speed)
- t456 > all t > t5
- dice loss > logcoshdice, focal (van muy lento, probar al final cuando tenga buenos modelos?)
- Adam > AdamW
- lr scheduler no mejora casi nada (usar al final con modelos grandes)
- eliminar masks con menos de 10 px mejora val pero submission se queda igual (postproc)
- encoders: resnest26d > resnet34 > resnet18

PROBAR:

- time series
	- pre, during, post (t456)
	- first, during, last (t158)
	- all
	- pre (t12345)
	- post (t5678)
- label smoothing
- encoders (resnest26d, resnext50_32d, efficientnet, convnextv2, ...) https://huggingface.co/docs/timm/results
- data augmentation (además de flips: cutout, cutmix, mixup, ...)
- decoders (fpn, deeplabv3, ...)
- threshold optimization (la gente está usando 0.4)
- postprocessing:
	- Contrails must contain at least 10 pixels
	- At some time in their life, Contrails must be at least 3x longer than they are wide
	- Contrails must either appear suddenly or enter from the sides of the image
	- Contrails should be visible in at least two image
- tta
- cv
- ensambling