Challenge: https://www.kaggle.com/c/siim-isic-melanoma-classification/

Este es el dataset que usamos: https://www.kaggle.com/sensioai/melanoma224. Contiene las imágenes del dataset original re-escaladas a 224x224 píxeles.

# Día 1

- Intro competi
- Descarga datos
- [Exploración datos](./exploracion.ipynb)
- Sample submission -> 0.5
- [Baseline Keras](./keras_baseline.ipynb): resnet50 congelada, cabeza lineal entrenada en subset 224px, adam 1e-3, batch size 64 -> 0.764
  
# Día 2

- Mejora dataset (más rápido), guardar mejor modelo durante el entrenamiento para cargar al final [aquí](./keras_baseline2.ipynb) -> 0.782
- Usamos todos los datos de train -> 0.844
- 3 fold cross validation, predicción ensamble [aquí](./keras_cv.ipynb): resnet50 congelada, cabeza lineal entrenada en subsets 224px, adam 1e-3, batch size 64 -> 0.811
- 5 fold cross validation, predicción ensamble, resnet50 congelada, cabeza lineal entrenada en subsets 224px, adam 1e-3, batch size 64 -> 0.827 (no hay notebook, pero lo he probado)

# Día 3

- Añadimos [data augmentation](./keras_data_augmentation.ipynb)
  - flip horizontal + flip vertical tf -> 0.7972
  - random flip keras -> 0.7873
- Añadimos [TTA](./keras_tta.ipynb) con flip horizontal y vertical -> 0.8060
- Usamos todos los datos de train -> 0.8485

# Día 4

- Añadimos [metadatos](./keras_meta.ipynb) en baseline con data augmentation -> 0.8179
- Todos los datos -> 0.8774

# Día 5

- [Pytorch baseline](./pytorch_baseline.ipynb) -> 0.7436

# Día 6

- [Data Augmentation](./pytorch_da.ipynb) -> 0.7209
- [Metadata](./pytorch_meta.ipynb) -> 0.7524

# Día 7

Se acabó la competición. Revisamos soluciones ganadoras. Conclusiones:

- Usar más datos si el reglamento lo permite
- Modelos ensamblados
- mucho TTA 
- Pytorch + GPU + amp > tensorflow + TPU
- EfficientNet B3-B7, se_resnext101, resnest101 > resnet
- si tenemos diferentes fuentes de datos, tener modelos en el ensamblado con y sin.
- con imágenes, diferentes modelos a diferentes resoluciones (de 512 para arriba si es posible)
- mucho DA
- hacer pipelines que permitan rápida experimentación
- si el dataset está desbalanceado, probar upsampling
- learning rate scheduling, Adam 3e-4
- re-entrenar backbone
- pseudolabeling
- weight averaging