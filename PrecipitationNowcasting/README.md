https://community.solafune.com/competitions/f87811b8-1964-4f4b-84b3-6fddd67ec4b1

Regresion a partir de series temporales de imágenes de satelites.

3 modalidades: himawari (16 bandas), goes (16 bandas), meteosat (16 bandas)
Tamaño inputs 81×81 (Himawari) / 144×144 (Meteosat) / 141×141 (GOES)

Para cada fecha podemos tener hasta 3 observaciones anteriores.

Depende de la zona, solo una modalidad está disponible (nunca usaremos más de una a la vez).

Ejemplo baseline: https://community.solafune.com/competitions/f87811b8-1964-4f4b-84b3-6fddd67ec4b1?tab=&menu=discussion&topicId=e26cbca1-bfd2-4e79-970e-79b63986ad32

# Data

### Satellite Band Mapping & Reference Alignment

To use pretrained models (like **SaTformer** trained on the 11 SEVIRI bands of Meteosat), we must physically align the 16 bands of **Himawari-8/9 (AHI)**, **GOES-16 (ABI)**, and **Meteosat-12 (FCI)** based on their central wavelengths ($\mu\text{m}$).

#### **11-Band Aligned Mapping (0-indexed)**
```python
BAND_MAPPING = {
    'himawari': [4, 6, 10, 11, 13, 14, 15, 2, 3, 7, 9],  # B05, B07, B11, B12, B14, B15, B16, B03, B04, B08, B10
    'goes':     [4, 6, 10, 11, 13, 14, 15, 1, 2, 7, 9],  # C05, C07, C11, C12, C14, C15, C16, C02, C03, C08, C10
    'meteosat': [6, 8, 11, 12, 13, 14, 15, 2, 3, 9, 10]   # nir_16, ir_38, ir_87, ir_97, ir_105, ir_123, ir_133, vis_06, vis_08, wv_63, wv_73
}
```

#### **Detailed Band-by-Band Alignment & Justification**

| SEVIRI Target | Target $\lambda$ ($\mu\text{m}$) | Himawari-9 Band (Index) | GOES-16 Band (Index) | Meteosat-12 Band (Index) | Physical Property / Application |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`IR_016`** | $1.64$ | `B05` (**4**) | `C05` (**4**) | `nir_16` (**6**) | Near-Infrared (Snow/Ice/Cloud phase) |
| **`IR_039`** | $3.92$ | `B07` (**6**) | `C07` (**6**) | `ir_38` (**8**) | Midwave Infrared (Shortwave window / Night fog) |
| **`IR_087`** | $8.70$ | `B11` (**10**) | `C11` (**10**) | `ir_87` (**11**) | Thermal Infrared (Cloud-top phase / Water vapor) |
| **`IR_097`** | $9.66$ | `B12` (**11**) | `C12` (**11**) | `ir_97` (**12**) | Thermal Infrared (Ozone absorption) |
| **`IR_108`** | $10.80$ | `B14` (**13**) | `C14` (**13**) | `ir_105` (**13**) | Thermal Infrared (Clean window / Cloud-top temp) |
| **`IR_120`** | $12.00$ | `B15` (**14**) | `C15` (**14**) | `ir_123` (**14**) | Thermal Infrared (Dirty window / Split-window) |
| **`IR_134`** | $13.40$ | `B16` (**15**) | `C16` (**15**) | `ir_133` (**15**) | Thermal Infrared ($\text{CO}_2$ absorption / Cloud height) |
| **`VIS006`** | $0.63$ | `B03` (**2**) | `C02` (**1**) | `vis_06` (**2**) | Visible (Red / Land cover / Aerosols) |
| **`VIS008`** | $0.81$ | `B04` (**3**) | `C03` (**2**) | `vis_08` (**3**) | Near-Infrared (Vegetation / Land-water contrast) |
| **`WV_062`** | $6.25$ | `B08` (**7**) | `C08` (**7**) | `wv_63` (**9**) | Thermal Infrared (Upper-level water vapor) |
| **`WV_073`** | $7.35$ | `B10` (**9**) | `C10` (**9**) | `wv_73` (**10**) | Thermal Infrared (Lower-level water vapor) |

#### **References**
* **Himawari-8/9 (AHI):** JMA Advanced Himawari Imager Spectral Response Functions (Japan Meteorological Agency, 2023).
* **GOES-16 (ABI):** NOAA GOES-R Advanced Baseline Imager Technical Summary Chart (NOAA/NASA, 2018).
* **Meteosat-12 (FCI):** WMO OSCAR Details for Instrument FCI / Flexible Combined Imager (WMO, 2023).

---

### Raw Satellite Data Formats

#### Himawari-8/9 - Data Format
The Himawari-8/9 Multi-Spectral Instrument, Level-1B image contains information on 16 bands and includes the following bands:
`'B01' (0.47 µm), 'B02' (0.51 µm), 'B03' (0.64 µm), 'B04' (0.86 µm), 'B05' (1.6 µm), 'B06' (2.3 µm), 'B07' (3.9 µm), 'B08' (6.2 µm), 'B09' (6.9 µm), 'B10' (7.3 µm), 'B11' (8.6 µm), 'B12' (9.6 µm), 'B13' (10.4 µm), 'B14' (11.2 µm), 'B15' (12.3 µm), 'B16' (13.3 µm)`.
The images have been processed from the Full disk image to the region of interest from their own perspective locations with their own unique datetimes.

#### GOES - Data Format
The GOES Multi-Spectral Instrument, Level-1B image contains information on 16 bands and includes the following bands:
`'C01' (0.47 µm), 'C02' (0.64 µm), 'C03' (0.86 µm), 'C04' (1.37 µm), 'C05' (1.6 µm), 'C06' (2.2 µm), 'C07' (3.9 µm), 'C08' (6.2 µm), 'C09' (6.9 µm), 'C10' (7.3 µm), 'C11' (8.4 µm), 'C12' (9.6 µm), 'C13' (10.3 µm), 'C14' (11.2 µm), 'C15' (12.3 µm), 'C16' (13.3 µm)`.
The images have been processed from the Full disk image to the region of interest from their own perspective locations with their own unique datetimes.

#### Meteosat - Data Format
The Meteosat Multi-Spectral Instrument, Level-1C image contains information on 16 bands and includes the following bands:
`'vis_04' (0.44 µm), 'vis_05' (0.51 µm), 'vis_06' (0.64 µm), 'vis_08' (0.86 µm), 'vis_09' (0.91 µm), 'nir_13' (1.38 µm), 'nir_16' (1.61 µm), 'nir_22' (2.25 µm), 'ir_38' (3.80 µm), 'wv_63' (6.30 µm), 'wv_73' (7.35 µm), 'ir_87' (8.70 µm), 'ir_97' (9.66 µm), 'ir_105' (10.50 µm), 'ir_123' (12.30 µm), 'ir_133' (13.30 µm)`.
The images have been processed from the Full disk image to the region of interest from their own perspective locations with their own unique datetimes.

# Ideas

SaTformer https://github.com/leharris3/satformer / https://arxiv.org/abs/2511.11090

Advanced Methodologies & Loss Functions:

To win this challenge, standard MSE loss is not enough. Because heavy rain is extremely rare (class imbalance), models trained purely on MSE will predict safe, blurry, low-intensity averages to minimize the squared error, which severely hurts performance on actual rain events.

### **Target Normalization & Discretization Strategy**
* **The Problem:** The `gpm_imerg` target rain rates are heavily right-skewed, ranging from $0.0$ up to $96.51\text{ mm/h}$. Over $90\%$ of the target pixels are $\le 0.42\text{ mm/h}$ (practically dry). If we normalize by simply dividing by $100$ and applying linear binning, over $90\%$ of the non-zero rain data is crammed into the first two bins (Bins 0 and 1), leaving the remaining 62 bins empty.
* **The Solution:** We apply a **Logarithmic Transformation** on the target variable inside the training module:
  $$y_{\text{norm}} = \frac{\ln(1 + y)}{4.58}$$
  This maps the physical range $[0.0, 96.51]$ to $[0.0, 1.0]$. This log-normalized target is then discretized into 64 ordinal bins, spreading the light, moderate, and heavy rain rates beautifully across all 64 bins.
* **Evaluation Metric Alignment:** During the forward pass, we calculate the continuous expected value in the log-normalized space:
  $$\hat{y}_{\text{norm}} = \sum_{i=0}^{63} P(\text{bin}_i) \cdot \text{value}(\text{bin}_i)$$
  To compute the exact competition metric (RMSE), we invert the log-transformation back to the original physical space (mm/h) before computing the squared error:
  $$\hat{y}_{\text{physical}} = \exp(\hat{y}_{\text{norm}} \cdot 4.58) - 1$$
  $$\text{RMSE} = \sqrt{\text{Mean}((\hat{y}_{\text{physical}} - y)^2)}$$

A. Discretization & Classification Losses (Focal / ML-Dice)
Instead of treating the problem purely as a regression task:
- Binning: Discretize the continuous rain rates into $N$ ordinal bins (e.g., no rain, light, medium, heavy, extreme).
- Loss Formulation: Use Focal Loss or Multi-Level Dice Loss to train the model to predict the probability distribution over these bins.
- Regression Mapping: Convert the predicted probabilities back to continuous values (e.g., by taking the expected value or using a learned projection layer).

Why it works: Focal Loss forces the model to focus on hard-to-predict, high-intensity rain events, preventing the "no-rain" majority pixels from dominating the gradient.

B. Importance Sampling & Dataset Refinement
- Satellite datasets are dominated by dry days. If you train on all available data, the model will overfit to "zero rain".
- Solution: Implement importance sampling during batch generation. 

Dealing with different satellites:

- Strategy A: Train 3 Separate Models (Recommended for Simplicity & Performance)
Since the test set locations map 1-to-1 to a specific satellite, you can easily train three separate models (one for Himawari, one for GOES, and one for Meteosat) and route the test samples accordingly.

Location to Satellite Mapping in your Dataset:
Himawari-9: kanto_region, mekong_delta, north_sumatra, northeast_malaysia, quang_nam, sri_lanka, sylhet
Meteosat-12: limpopo_province, lombardia, maputo_province, niger_state, sofala_province, tanganyika, valencia
GOES-16: mexico, peru, rio_grande_do_sul, upper_midwest
Pros:
No Alignment Needed: Each model natively learns the exact spectral profile, calibration, and viewing geometry of its respective satellite.
Regional Specialization: Weather patterns and climates are highly region-specific (e.g., tropical monsoons in Southeast Asia vs. midlatitude storms in the US Midwest). A dedicated model can specialize in its region's climatology.
Higher Accuracy: Historically, in transfer-learning competitions, models trained on a single satellite's domain outperform models forced to generalize across unaligned domains.
Cons:
You have to train and maintain 3 models (though they can share the exact same code and architecture, just trained on different subsets of the data).

- Strategy B: Train a Single Unified Model (Recommended for Data Volume & Generalization)
If you want to train a single model to leverage the combined size of the entire dataset, you must align the inputs first.

How to Align the Inputs:
Spectral Band Re-indexing (Mapping): Re-order the channels of each satellite so that they physically match. Since GOES lacks a green band and Himawari lacks a cirrus band, you should select a common subset of physically matched channels (e.g., 10 or 11 aligned bands, similar to the 11-band SEVIRI format in Weather4cast):
Example Common Subset: Blue (0.47 μm), Red (0.64 μm), Near-IR (0.86 μm), SWIR (1.6 μm), Midwave IR (3.9 μm), Water Vapor (6.2 μm, 7.3 μm), and the Window IR channels (10.4 μm, 11.2 μm, 12.3 μm, 13.3 μm).
Physical Calibration: Instead of feeding raw uint8 values, convert them to physical units:
Reflectance (0.0 to 1.0) for visible and near-infrared bands.
Brightness Temperature (Kelvin, typically 180K to 320K) for thermal infrared bands.
If the exact calibration coefficients are not available, apply satellite-specific channel normalization (subtracting the mean and dividing by the standard deviation calculated per channel, per satellite over the training set).
Satellite / Region Embeddings: Pass a one-hot satellite ID or region embedding as an auxiliary input to your network (e.g., via FiLM layers or concatenating it to the latent features). This tells the model which sensor's viewing geometry and which regional climate it is processing.

Summary Recommendation

Start with Strategy A (3 Separate Models): It is the fastest way to build a highly accurate baseline without worrying about complex cross-calibration and band-mapping bugs.
Move to Strategy B (Unified Model) later if you feel your individual models are overfitting due to a lack of data, and you want to leverage the transfer-learning capabilities of a larger, unified dataset.

# Report

Satformer with bin classification loss, 32x32 4 frames.

Baseline: best val_rmse 0.7698 tras 3 entrenos de baseline (puedde seguir mejorando pero no mucho más como para llegar a 0.6)
- 1.279 / 1.014
- 0.8217 / 0.934

Scratch:
- 32x32 3 frames running ...

## TODO

- train from scratch
  - 32x32 3 frames
  - 64x64 3 frames (BS 16)
- data augmentation 
- tta for inference