# Finding Mining Sites

https://solafune.com/competitions/58406cd6-c3bb-4f7a-85c7-c5a1ad67ca03?menu=lb&tab=&modal=%22%22

## Experiments

baseline rgb -> 0.83495 / 0.8247 (overfit a training)
flips cropt lr sch -> 0.875 / 0.8148 (overfit a validation)


## Ideas

- [ ] Data augmentation (rotation, flips, random resized, crops, etc)
- [ ] Explorar combinación de bandas / índices (fc, ndvi, ndwi, ...)
- [x] Diferentes modelos (resnet, efficientnet, ...)
	- resnet18, 34, 50 (resnest, se_resnext)
	- efficientnet b3, b5, b7
	- mit (vision transformer) b1, b2, b4 -> no acepta más de 3 canales a la entrada
- [ ] Modelo NASA/IBM https://huggingface.co/ibm-nasa-geospatial
- [ ] lr/seed hpopt
- [ ] lr scheduling
- [x] threshold tuning 
- [ ] tta 
- [ ] ensambles
- [ ] cross validation
- [ ] train with val 