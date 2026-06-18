## SaTformer: Quick Start & Customization Guide

### 1. Repository & Pre-trained Weights

- **GitHub:** [leharris3/satformer](https://github.com/leharris3/satformer)
- **Pre-trained Weights (HuggingFace Hub):** `sf-64-cls.pt`
- **ArXiv Paper:** [A Space-Time Transformer for Precipitation Nowcasting (arXiv:2511.11090)](https://arxiv.org/abs/2511.11090)

---

### 2. Using the Official Implementation

#### **A. Download Pre-trained Weights**

Use `huggingface_hub` to download the model weights:

```python
from huggingface_hub import hf_hub_download

weights_path = hf_hub_download(
    repo_id="leharris3/satformer",
    filename="sf-64-cls.pt",
    local_dir="weights"
)
print(f"Weights downloaded to: {weights_path}")
```

#### **B. Instantiate and Run the Model**

Typical setup for Weather4cast 2025 workflow:

```python
import torch
from src.model.SaTformer.SaTformer import SaTformer

model = SaTformer(
    dim=512,             # Embedding dimension
    num_frames=4,        # e.g. 1 hour of 15-min slots
    num_classes=64,      # Number of discrete precipitation bins
    image_size=32,       # Input spatial HxW
    patch_size=4,        # (32x32 / 4x4)
    channels=11,         # Satellite radiance bands
    depth=12,            # Transformer encoder blocks
    heads=8,             # Attention heads
    dim_head=64,         # Head dimension
    attn_dropout=0.1,
    ff_dropout=0.1,
    rotary_emb=False,    # Use learnable positional embeds
    attn="ST^2"          # Full Space-Time Attention
)
model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
model.eval()

# Example input: (batch, frames, channels, height, width)
x = torch.rand(1, 4, 11, 32, 32)
with torch.no_grad():
    logits = model(x)
print("Output logits shape:", logits.shape)  # torch.Size([1, 64])
```

---

### 3. Adapting SaTformer for Solafune ($41 \times 41$ regression)

Key differences in the Solafune challenge compared to the standard Weather4cast 2025 setup:
- **16-band imagery** (instead of 11 bands).
- **Up to 3 previous observations** (at $t-30$, $t-20$, and $t-10$ minutes), meaning `num_frames=3`.
- **Grid-level prediction** (the target is a $41 \times 41$ spatial grid of precipitation rates, rather than a single scalar cumulative rainfall value).

Here is how you can adapt SaTformer's architecture for your $41 \times 41$ grid regression task:

#### **Adaptation Option A: SaTformer as a Feature Extractor (Encoder-Decoder)**

Instead of splicing out only the `[CLS]` token for a single classification value, you can use SaTformer's transformer layers to output a sequence of spatial-temporal tokens, and then decode them back into a $41 \times 41$ grid:

1. Modify the input channels to 16 to match Himawari/GOES/Meteosat.
2. Extract the spatial-temporal token embeddings from SaTformer's final layer.
3. Reshape the tokens back into a 3D spatial-temporal grid: `(batch, frames, dim, H_patches, W_patches)`.
4. Pass the grid through a convolutional decoder (e.g., a U-Net decoder) to upsample to $41 \times 41$.

#### **Adaptation Option B: Reformulating to Multi-Class Pixel Classification**
Following SaTformer's core winning methodology, you can discretize the $41 \times 41$ grid of continuous rain rates into 64 bins per pixel.

Your target becomes a shape of (batch, 41, 41, 64).
Your model predicts a probability distribution for each pixel in the $41 \times 41$ grid.
During inference, you take the expected value across the 64 bins for each pixel to get the continuous precipitation rate: 

$$\hat{y}_{\text{pixel}} = \sum_{i=0}^{63} P(\text{bin}_i) \cdot \text{value}(\text{bin}_i)$$

#### **4. Key Takeaways to Replicate Their Victory**
If you use this implementation, make sure to include these two crucial components from their paper:

Class-Weighted Cross-Entropy Loss: Since "no rain" represents the vast majority of pixels, calculate the relative frequency of each bin in your training set and weight the loss using: 

$$w_i = -\log\left(\frac{|\mathcal{D}_i|}{|\mathcal{D}_{\text{total}}|}\right)$$
Full Space-Time Attention (S + T): Do not use decoupled space-then-time attention (which is common in video models to save memory). Because the satellite frames are short (3 frames) and the spatial resolution is small (e.g., $32 \times 32$ or $41 \times 41$), full 3D attention is highly computationally feasible and yields significantly better results (as shown in their ablation studies).