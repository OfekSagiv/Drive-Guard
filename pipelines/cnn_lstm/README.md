# Pipeline: CNN + LSTM

Spatial feature extraction using a **CNN backbone**, followed by an **LSTM** temporal classifier.

This pipeline will follow the same general structure as the other pipelines in this repo:
1. Use the shared preprocessing outputs (`ds_driveguard_temporal_roi/`)
2. Train a CNN spatial model to extract per-frame feature vectors
3. Extract features for each 16-frame sequence → `.npy` tensors
4. Train an LSTM on the feature sequences for activity classification
5. Evaluate on the test split and compare against the ViT-Transformer baseline

---

## Step 1 — Spatial Model: EfficientNet-B4

📓 **Notebook**: [efficientnet_b4_cnnV2.ipynb](https://drive.google.com/file/d/1d-nWFxsMm_CZGMuc2BUvVTKaUBM8w9IZ/view?usp=sharing)

Fine-tunes a pretrained **EfficientNet-B4** (380×380 input) on the 3-class driver behavior dataset: **Drink**, **Phone**, **Safe**.

### Model & Configuration

| Parameter | Value |
|---|---|
| Backbone | `efficientnet_b4` (timm) |
| Input size | 380×380 |
| Batch size | 64 |
| Classes | Drink, Phone, Safe |

### Training Phases

**Phase 1 — Warmup (head only)**
Backbone is fully frozen. Only the classifier head is trained with `OneCycleLR` (`lr=1e-3`) for 5 epochs. Prevents random head initialization from corrupting pretrained weights.

**Phase 2 — Full Fine-Tuning**
All layers unfrozen with differential learning rates:
- Backbone (`conv_stem` + `blocks`) → `lr = 3e-5`
- Head (`classifier`) → `lr = 1e-4`

Scheduler: `CosineAnnealingLR`. Early stopping monitors **macro F1** (patience=8).

### Augmentation
- RandomHorizontalFlip, ColorJitter, RandomAffine, GaussianBlur, RandomGrayscale
- **Mixup** (`α=0.4`) + **CutMix** (`α=0.5`) at batch level

### Loss
`FocalLoss` (`γ=1.5`) with label smoothing (`0.1`). `WeightedRandomSampler` balances class frequency per batch.

### Dataset
`NestedROIDataset` — recursively scans class folders with `rglob('*.jpg')`, compatible with camera sub-directory structure from `extract_spatial_roi_ds.py`.

### Output
Best checkpoint saved to `MyDrive/DriveGuard/models/efficientnet_b4_spatial_model_v1.pth`.
