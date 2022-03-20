baseline
r18 10e 100 batches -> 0.84274/0.83410
r18 bs512 Adam1e-3 -> 0.74773/0.72305 (overfit en epoch 2)
r18 bs512 fp16 Adam1e-3 rgbnir -> 0.73979/0.72087 (hasta epoch 4 con bs1024 pero luego paso a 512 porque peta)
r18 rgnir -> 0.73669/0.72878
r18 nirgb -> probar
r50 rgbnir
(from scratch da peor)

RGBNirBIO
r18 256-512 -> 0.71825/0.69522
r18 100 -> running...

da
r18 rgbnir (flips) -> 0.728839

probar:

- añadir imágenes adicionales (nir en canal 4 con rgb, altitud y landcover en custom CNN feature extractor)
- añadir metadatos (país (1), environmental (27, hay que hacer imputing y standarization)) con mlp head
- transformer head ?
- data augmentation
- modelos más grandes
- lr scheduling
- tta
- ensambling

conclusiones:

rgbnir > rgb > rgnir
