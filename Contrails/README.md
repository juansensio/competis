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
unet resnest26d fc t456 da flips e100 lrsch -> 0.61026 / 0.602 (al tracear baja) / 0.601
unet r18 fc t5 da flips+resize512 e300 -> running ...


RESULTADOS:

- añadir da (flips) mejora
- false color > all bands(mean_std) > all_bands (min_max) (metric & speed)
- t456 ~ t12345 > t158 > all t ~ t5678 > t5 (probar t345, segun paper funciona mejor)
- dice loss > logcoshdice, focal (van muy lento, probar al final cuando tenga buenos modelos?)
- Adam > AdamW
- lr scheduler no mejora casi nada (usar al final con modelos grandes)
- eliminar masks con menos de 10 px mejora val pero submission se queda igual (postproc)
- encoders: resnet34 > resnest26d (traced) > resnet18
- 512 > 256 

PROBAR:

- encoders (resnet50, resnext50_32d, efficientnet, convnextv2, ...) https://huggingface.co/docs/timm/results
- label smoothing
- data augmentation (además de flips: cutout, cutmix, mixup, ...)
    Data augmentation is applied during training to reduce overfitting:
    a scale factor is chosen uniformly randomly at each iteration,
    and then the image is scaled according to the scale factor, and
    a crop of the scaled image is used for training. We found this
    significantly helps overall performance. (en paper suben hasta 1024)
- decoders (fpn, deeplabv3, ...)
- tta
- cv
- ensambling


