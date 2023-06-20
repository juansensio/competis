baseline r18 t5 e15 -> 0.43907 / 0.427 (test para probar, parece que la metrica se parece en val y test, así que guay)
la baseline tenia un bug en el decoder de la unet, repetir
baseline r18 t5 e100 -> 0.47794
unet r18 t5 da flips e200 -> 0.54259
unet r18 t456 da flips e200 -> 0.57122
REFACTOR
unet r18 fc t5 da flips e100 -> 0.567
unet r18 all_bands t5 da flips e100 min_max -> mal
unet r18 all_bands t5 da flips e100 mean_std -> 0.54979
unet r18 fc t456 da flips e100 -> running...
unet r18 fc t456 da flips e200 lrsch -> next
unet r18 fc t456 da flips e200 lrsch AdamW -> next
unet r18 fc t456 da flips e200 lrsch AdamW resnet34 -> next
unet r18 fc t456 da flips e200 lrsch AdamW resnest26d -> next

RESULTADOS:

- añadir da mejora
- usar t456 mejora
- false color > all bands(mean_std) > all_bands (min_max) (metric & speed)
- dice loss > logcoshdice, focal (van muy lento)

PROBAR:

- lr scheduler
- optimizer (AdamW)
- encoders (resnest26d, resnext50_32d, efficientnet, ...) https://huggingface.co/docs/timm/results
- loss smooth 1
- time series
	- pre, during, post (t456)
	- all
- data augmentation (además de flips: cutout, cutmix, mixup, ...)
- decoders (fpn, deeplabv3, ...)
- threshold optimization
- tta
- ensambling
