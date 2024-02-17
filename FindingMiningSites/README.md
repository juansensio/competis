# Finding Mining Sites

https://solafune.com/competitions/58406cd6-c3bb-4f7a-85c7-c5a1ad67ca03?menu=lb&tab=&modal=%22%22

## Experiments

baseline rgb -> 0.83495 / 0.8247 (overfit a training)
flips cropt lr sch -> 0.875 / 0.8148 (overfit a validation)
seresnext -> 0.929 / 0.868
maxvit base rgb -> 0.94 / 0.931 (BEST)
convnextv2, eva02 -> no aprenden
maxvit base fc -> 0.943 / 0.903
maxvit base all bands -> 0.90 / 0.8
maxvit base rgb + ndvi -> 0.93 / 0.9
maxvit base rgb + indices -> 0.92 / 0.871
maxvit large rgb -> 0.93 / 0.915

## Ideas

- [x] Data augmentation (rotation, flips, random resized, crops, etc)
- [x] Explorar combinación de bandas / índices (fc, ndvi, ndwi, ...)
	- rgb > el resto (probablemente porque los modelos tochos han sido entrenados con rgb y hay muy pocos datos para ft con más bandas/indices)
- [x] Diferentes modelos (resnet, efficientnet, ...) https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv
- [ ] Modelo NASA/IBM https://huggingface.co/ibm-nasa-geospatial
- [ ] lr/seed hpopt
- [x] lr scheduling
- [x] threshold tuning 
- [ ] tta 
- [ ] ensambles
- [ ] cross validation
- [ ] train with val 