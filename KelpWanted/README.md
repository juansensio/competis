# Kelp Wanted: Segmenting Kelp Forests

https://www.drivendata.org/competitions/255/kelp-forest-segmentation/page/791/

## Experiments

Baseline: 0.47194 / 0.4893
	- rgb, no da, default hparams
Feature engineering: r18, no da, default hparams (he probado varias normalizaciones y con/sin índices y esta es la mejor combinación)
	- false color (norm 2e4) + ndvi + ndwi: 0.651 / 0.6586 
Data augmentation: fci, default hparams 
	- flips + rots + resized random crops: 0.643 (no mejora... quizás si en modelos más grandes?)
FCI con máscara en loss: no da, default hparams
	- dem + clouds binary loss mask: 0.661 / 0.6705 (BEST)
Encoders
	- resnet34: 0.650
	- resnet50: 0.661
	- resnest50d: 0.668 (BEST)
	- se_resnext (no va el link de descarga)
	- efficientnet b3: 0.667 
	- efficientnet b5: 0.664
Architectures: fcim
	- unet++ (resnest50d): 0.678 (BEST)
	- pan, deeplabv3 (no tira con las backbones)
Volver a probar data augmentation, debería mejorar...: 
	- flips + rotate + crops: 0.664 (no mejora)
	- flips + rotate: 0.670
hpopt con unet++ resnest50d
Loss
	- focal


## Ideas

- [x] Explorar combinación de bandas / índices (ndvi, ndwi)
	- False Color: NIR, Red, and Green (enhances vegetation)
	- NDVI: This index is commonly used to detect live green vegetation and can be useful to distinguish kelp from non-vegetative elements.
	- NDWI: This index helps in enhancing the presence of water features while suppressing vegetation and soil noise, potentially aiding in the isolation of kelp areas.
- [x] Data augmentation (rotation, flips, random resized, crops, etc)
- [x] Uso de DEM / nubes como máscaras en la loss (para que no haga caso a esos pixeles)
	- En muchas imágenes hay pixeles de DEM/cloud que hacen overlap con ground truth...
- [ ] Diferentes modelos (unet, deeplabv3, ...) y encoders (resnet, efficientnet, ...)
	- Encoders
		- resnet18, 34, 50 (resnest, se_resnext)
		- efficientnet b3, b5, b7
		- mit (vision transformer) b1, b2, b4 -> no acepta más de 3 canales a la entrada
	- Architectures
		- unet, unet++
		- pan
		- deeplabv3 (deeplabv3+)
- [ ] Diferentes loss functions (dice, focal, ...)
- [ ] lr/seed hpopt
- [ ] lr scheduling
- [ ] threshold tuning
- [ ] tta
- [ ] ensambles
- [ ] postprocessing (erosion, dilation, etc) morphological operations
- [ ] train with val

error analysis:
- [ ] track other metrics: if precision is low, you might focus on reducing false positives, potentially by adjusting post-processing thresholding or refining the model to better distinguish kelp from similar features
	- precision: high false positives (low precision). High precision means that when your model predicts kelp, it's likely to be correct.
	- recall: false negatives (low recall). High recall means your model is effectively capturing most of the kelp areas.
	- iou: ensuring that your model is accurate not only in detecting kelp presence but also in predicting the correct size and position of the kelp patches.
- [ ] visualize the model's predictions on the val set to get a sense of where it's doing well and where it's struggling
- [ ] confusion matrix analysis
- [ ] band importance and index sensitivity analysis