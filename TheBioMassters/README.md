# The BioMassters

https://www.drivendata.org/competitions/99/biomass-estimation/

Predict [Aboveground biomass](https://www.un-redd.org/glossary/aboveground-biomass) (ABGM) from Sentinel 1 and Sentinel 2 image time series using LiDAR and in-situ ground truth.

## Data

The feature data for this challenge is imagery collected by the Sentinel-1 and Sentinel-2 satellite missions for nearly 13,000 patches of forest in Finland. Each patch (also called a "chip") represents a different 2,560-by-2,560-meter area of forest. The data were collected over a period of 5 years between 2016 and 2021.

Each label in this challenge represents a specific chip, or a distinct area of forest. LiDAR measurements are used to generate the biomass label for each pixel in the chip. For each chip, a full year's worth of monthly satellite images for that area are provided, from the previous September to the most recent August.

For each chip, there should be 24 images in total that can be used as input - 12 for each satellite source. However, note that there are some data outages during this time period, so not all chips will have coverage for every month.

All of the satellite images have been geometrically and radiometrically corrected and resized to 10 meter resolution. Each resulting image is 256 by 256 pixels, and each pixel represents 10 square meters. Images represent monthly aggregations and are provided as GeoTIFFs with any associated geolocation data removed.

The ground truth values for this competition are yearly Aboveground Biomass (AGBM) measured in tonnes. Labels for each patch are derived from LiDAR (Light Detection and Ranging), a remote sensing technology that provides 3D information about the terrain and vegetation. The label for each patch is the peak biomass value measured during the summer.

Similarly to the feature satellite imagery, LiDAR data is provided as images that cover 2,560 meter by 2,560 meter areas at 10 meter resolution, which means they are 256 by 256 pixels in size

## Prediction

You only need to generate one biomass prediction per chip, but can use as many of the chip's multi-temporal (different months) or multi-modal (Sentinel-1 or Sentinel-2) satellite images as you like. Predictions should include a yearly peak aboveground biomass (AGBM) value for each 10 by 10 pixel in the chip.

## Metric

To measure your model’s performance, we’ll use a metric called Average Root Mean Square Error (RMSE). RMSE is the square root of the mean of squared differences between estimated and observed values. RMSE will be calculated on a per-pixel basis (i.e., each pixel in your submitted tif for a patch will be compared to the corresponding pixel in the ground-truth tif for the patch). RMSE will be calculated for each image, and then averaged over all images in the test set. This is an error metric, so a lower value is better.

## Submission

You must submit your predictions in the form of single-band 256 x 256 tif images, zipped into a single file. The lone band should contain your predictions for the yearly AGBM for that chip.

The format for the submission is a .tar, .tgz, or .zip file containing predicted AGBM values for each chip ID in the test set. Each tif must be named according to the convention {chip_id}\_agbm.tif. The order of the files does not matter. In total, there should be 2,773 tifs in your submission.

## Approach

### Baseline

Get familiar with the data, metric and submission process.

UNet on S2 RGB single image (April) -> 41.517
UNet on S1 VVVH single image (April) -> 45.232

### Models

UNet S1 VVVH + S2 RGB ingle image (April) da -> 37.865
UNet S1 VVVH + S2 RGB (full time series) da concat output -> 34.353
UNet S1 VVVH + S2 RGB (full time series) da concat features -> 31.982 / 32.1235 (31.8992 bck, viejo da mejor?)
UNet S1 VVVH + S2 RGB NDVI NDWI Clouds (full time series) da concat features -> 31.197
UNet S1 VVVH + S2 RGB NDVI NDWI Clouds (full time series) da2 concat features -> 30.384 / 30.488 (definitivamente da viejo peor)
UNet S1 VVVH + S2 RGB NDVI NDWI Clouds (full time series) da2 concat features no val scheduler ms -> 28 / 30.2151
UNet S1 + S2 (full time series, all bands, indices) da2 ltae scheduler ms -> 29.6
UNet S1 + S2 (full time series, all bands, indices) da2 concat features scheduler ms -> muy lento
UNet S1 VVVH + S2 RGB NDVI NDWI Clouds da2 ltae scheduler oc -> 29.48 (28.879 tta) / 29.6418 29.079 (tta)
UNet S1 VVVH + S2 RGB NDVI NDWI Clouds da2 ltae scheduler oc no val -> 27.6 (tta) / 29.3775 (tta) no mejora
UNet S1 VVVH + S2 RGB NDVI NDWI Clouds da2 concat features scheduler oc -> 29.71 (29.0516 tta)
ensamble ltae+concat features -> 28.645 (tta) / 28.8447 (tta) (best)
UNet S1 VVVH + S2 RGB NDVI NDWI Clouds da2 ltae scheduler oc se_resnext30_32x4d -> 29.425
ensamble ltae+concat+ltae r50 -> 28.655 (tta) no mejora
UNet all bands da2 ltae scheduler oc -> 29.623
