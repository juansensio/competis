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
unet seresnextaa101d_32x8d fc t5 da flips+resize512 e100 -> 0.58112	 / 0.62321 / 0.605 (seguir entrenando)
unet seresnextaa101d_32x8d fc t5 da flips+resize512 e200 SGD lrsch 0.2 bs256 -> no mejoraba el anterior
unet efficientnet_b0 fc t5 da (flips+crops) e100 bs64 -> 0.58909
unet efficientnet_b0 fc t5 da (flips+crops) e300 bs64 512 -> 0.618 / 0.628 / 0.594 (podría seguir aprendiendo, pero muy lento)
REFACTOR 2: en modelo, añadir extra conv para upsample final y mantener numero de filtros en convt del decoder
unet resnet34 fc t5 e30  -> 0.56470
unet resnet34 fc t456 e30  -> 0.58335
unet resnet34 fc t345 e30  -> 0.58359
unet resnet50d fc t5 384 lr 5e-4 AdamW -> 0.59095
unet resnet50d fc t345 384 lr 5e-4 AdamW -> 0.59311
unet resnest101e 384 AdamW Cosine sch lr 1e-3 30 epochs (0.628/0.638) -> running...
unet regnety_120 384 AdamW Cosine sch lr 1e-3 30 epochs -> next
unet resnest50d fc t5 384 sch lr 1e-3 30 epochs AdamW -> next
unet resnest101e 384 AdamW Cosine sch lr 1e-3 30 epochs t345 -> next

seresnextaa101d_32x8d
efficientnet_b0
tf_efficientnet_b7
convnextv2_base
maxvit_base_tf_512

lo que mejor de
unet resnet34 fc t345 512 da filps e200  -> next
unet resnet34 fc t345 512 da filps+crops e200  -> next

M: resnest101e 384 AdamW sin augmentations Cosine sch lr 1e-3 20 epochs (0.628/0.638)
A: resnest50d 384 lr 5e-4 bs 48 15 epochs optimizer AdamW sin augmentations (0.623) -> probar

RESULTADOS:

- false color > all bands(mean_std) > all_bands (min_max) (metric & speed)
- t345 ~t456 ~ t12345 > t158 > all t ~ t5678 > t5 
- 512 > 384 > 256 
- eliminar masks con menos de 10 px mejora val pero submission se queda igual (postproc)


PROBAR:

- encoders (features_only, traceables, < 100M params) https://huggingface.co/docs/timm/results

model, position, size 
maxvit_base_tf_512, 31, 100
convnextv2_base, 47, 88
tf_efficientnet_b7, 79, 66
seresnextaa101d_32x8d, 87, 100


- pseudolabelling (modelo en t5 para anotar t4, t6 y reenetrenar)
- tta
- ensambling
- cv
- train with val
- label smoothing
- more da (color, blur, cutout, mixup, cutmix, mosaic, mixup mosaic, mixup mosaic cutout, ...)
- decoders (bifpn)

https://arxiv.org/pdf/2304.02122.pdf 

We train the network using stochastic gradient descent with
10,000 iterations, using a batch size of 256 and a momentum
of 0.9. -> 20529 / 256 ~ 80 batches (125 epochs)
The learning rate is set to 0.2 and gradually decays to zero with half-cycle
cosine learning rate decay. To improve training stability, we
employ a linear warm up learning rate schedule, increasing
the learning rate linearly from 0 for the first 500 iterations (6 epochs).
L2 weight decay is used with a factor of 0.0001.

CONCLUSIONES para otros challenges:

- feature engineering
- hparam opt
- error analysis