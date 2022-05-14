baseline
r18 10e 100 batches -> 0.84274/0.83410
r18 bs512 Adam1e-3 -> 0.74773/0.72305 (overfit en epoch 2)
r18 bs512 fp16 Adam1e-3 rgbnir -> 0.73979/0.72087 (hasta epoch 4 con bs1024 pero luego paso a 512 porque peta)
r18 rgnir -> 0.73669/0.72878
r18 nirgb -> probar

RGBNirBio
r18 256-512 -> 0.71825/0.69522
r18 100 -> 0.71528/0.70231

RGBNirBioCountry
r18 -> 0.71052/0.70013
r18 (checkpoint val_error) ->0.704982/0.69495
r34 -> running ...

da
r18 rgbnir (flips) -> 0.728839

all r18 -> 0.7084 / 0.68894 (best) 0.68567 (tta)
all no val 5 epochs -> 0.62104430 (obvio, val in train) / 0.68840
all r34 -> 0.70467 / subir submissions
all resnest14d -> running ...

probar:

- añadir imágenes adicionales (nir en canal 4 con rgb, altitud y landcover en custom CNN feature extractor)
- transformer head ?
- data augmentation
- modelos más grandes
- lr scheduling
- tta
- ensambling

conclusiones:

rgbnir > rgb > rgnir
bio+country > bio > rgbnir
monitor val_error > monitor val_loss

da > no da
