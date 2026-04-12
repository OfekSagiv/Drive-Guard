# Pipeline: CNN + LSTM

Spatial feature extraction using **EfficientNet-B4** (1792-dim embeddings), followed by a **Bidirectional LSTM** temporal classifier.

---

## Architecture

| Stage | Model | Input | Output |
|---|---|---|---|
| Spatial | EfficientNet-B4 (`efficientnet_b4`, timm) | 380×380 ROI crops | 1792-dim feature vector |
| Feature extraction | EfficientNet-B4 with classifier head removed | 16-frame sequence | `(16, 1792)` .npy tensor |
| Temporal | Bidirectional LSTM (2 layers, 512 hidden) + linear head | `(16, 1792)` | Safe / Drink / Phone |

---

## Run Order

### Prerequisites
- Preprocessing done: `ds_driveguard_temporal_roi/` exists at repo root
- Google Drive folder: `MyDrive/DriveGuard/`
- All notebooks run on Google Colab (GPU required)

---

### Step 1 — Train spatial model (`efficientnet_b4_cnnV2.ipynb`) — [![Open in Drive](https://img.shields.io/badge/Open-Google%20Drive-blue)](https://drive.google.com/file/d/1d-nWFxsMm_CZGMuc2BUvVTKaUBM8w9IZ/view?usp=sharing)

**Input:** `ds_driveguard_spatial_roi/` uploaded as zip to `MyDrive/DriveGuard/`

- EfficientNet-B4 loaded via `timm`, 3-class classifier head
- Phase 1: freeze backbone, train head only with `OneCycleLR` (`lr=1e-3`) for 5 epochs
- Phase 2: unfreeze all layers, differential LRs (`backbone=3e-5`, `head=1e-4`), `CosineAnnealingLR`, early stopping on macro F1 (patience=8)
- Augmentation: flip, color jitter, affine, Gaussian blur, grayscale, MixUp (α=0.4), CutMix (α=0.5)
- Class imbalance: `WeightedRandomSampler` + FocalLoss (γ=1.5) with label smoothing (0.1)
- Dataset: `NestedROIDataset` — recursively scans class folders via `rglob('*.jpg')`

**Output:** `MyDrive/DriveGuard/models/efficientnet_b4_spatial_model_v1.pth`

---

### Step 2 — Extract features (`efficientnet_b4_photo_to_tensor.ipynb`) — [![Open in Drive](https://img.shields.io/badge/Open-Google%20Drive-blue)](https://drive.google.com/file/d/1H_qKe3CDfUtZZUP5HA3oFeggzsGXwEs7/view?usp=sharing)

**Input:**
- `MyDrive/DriveGuard/models/efficientnet_b4_spatial_model_v1.pth`
- `ds_driveguard_temporal_roi/` uploaded as zip to `MyDrive/DriveGuard/`

- Loads trained spatial model, removes classifier head (`reset_classifier(0)`)
- ImageNet normalization, resize to 380×380
- Per sequence: 16 frames → `(B×16, C, H, W)` → EfficientNet → reshape to `(16, 1792)` → save as float32 .npy
- Safe to re-run — skips already-extracted files

**Output:** `MyDrive/DriveGuard/CNN_Features/{train,val,test}/{class}/{seq_id}.npy` — shape `(16, 1792)`

17,019 total sequences: 10,259 train / 2,713 val / 4,047 test

---

### Step 3 — Train temporal model (`lstm_temporal_head.ipynb`) — [![Open in Drive](https://img.shields.io/badge/Open-Google%20Drive-blue)](https://drive.google.com/file/d/1orLB3SNmKI8qPM6lpi1uXgsSY5hxDbhF/view?usp=sharing)

**Input:** `MyDrive/DriveGuard/CNN_Features/{train,val,test}/{class}/a_column_co_driver_*.npy`

- Ablation study across 8 configs (dropout, noise_std, phone class weight, focal loss γ); winner selected by Phone F1, tiebroken by macro F1
- Full retrain of winning config:
  - Phase 1 head warmup: 5 epochs, encoder frozen (`lr=1e-3`)
  - Phase 2 fine-tune: differential LRs (encoder `2e-5`, head `5e-5`), CosineAnnealingLR, early stopping (patience=10)
- Architecture: input projection 1792→512 (Linear + LayerNorm + GELU), 2-layer bidirectional LSTM → 1024-dim → linear head
- Gaussian noise injection (`noise_std=0.03`) during training
- Loss: FocalLoss or CrossEntropyLoss with per-class weights + label smoothing (0.1)

**Output:** `MyDrive/DriveGuard/models/lstm_temporal_head_model.pth`

---

### Step 4 — Run inference (`infer.py`)

Weights and sample video are **auto-downloaded from Google Drive** on first run:

```bash
cd pipelines/cnn_lstm
python infer.py
```

Or with custom paths:

```bash
python infer.py \
    --video            /path/to/video.mp4 \
    --spatial_weights  /path/to/efficientnet_b4_spatial_model_v1.pth \
    --temporal_weights /path/to/lstm_temporal_head_model.pth \
    --output_video     ./out.mp4
```

Device auto-detected: Apple Silicon MPS (FP16) → CUDA (FP16) → CPU (FP32)

**Real-time performance** (measured on sample video, Apple Silicon):

| Metric | Value |
|---|---|
| Avg spatial inference | ~22 ms / sampled frame |
| Avg temporal inference | ~1.8 ms / window |
| Real-time budget | 200 ms / 6 frames |
| Processing speed | ~99 fps |
| Real-time factor | 3.3× faster than real-time ✓ |

---

## Google Drive Layout

```
MyDrive/DriveGuard/
├── models/
│   ├── efficientnet_b4_spatial_model_v1.pth
│   └── lstm_temporal_head_model.pth
├── all_cams_ds_driveguard_spatial_roi.zip
├── all_cams_ds_driveguard_temporal_roi.zip
└── CNN_Features/
    ├── train/
    ├── val/
    └── test/
```
