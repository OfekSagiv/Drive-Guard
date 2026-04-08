# Pipeline: ViT + Transformer

Spatial feature extraction using **ViT-SO400M-SigLIP** (1152-dim embeddings), followed by a **Transformer encoder** temporal classifier.

---

## Architecture

| Stage | Model | Input | Output |
|---|---|---|---|
| Spatial | ViT-SO400M (`vit_so400m_patch14_siglip_384`) | 384×384 ROI crops | 1152-dim feature vector |
| Feature extraction | ViT with classifier head removed | 16-frame sequence | `(16, 1152)` .npy tensor |
| Temporal | Transformer encoder + linear head (4 layers, 8 heads, 768 dim) | `(16, 1152)` | Safe / Drink / Phone |

---

## Run Order

### Prerequisites
- Preprocessing done: `ds_driveguard_temporal_roi/` exists at repo root
- Google Drive folder: `MyDrive/DriveGuard/`
- All notebooks run on Google Colab (GPU required)

---

### Step 1 — Train spatial model (`ViT_spatial_model.ipynb`)

**Input:** `ds_driveguard_spatial_roi/` uploaded as zip to `MyDrive/DriveGuard/`

- ViT-SO400M loaded via `timm`, 3-class classifier head
- Phase 1: freeze backbone, train head only (`lr=1e-3`)
- Phase 2: unfreeze deep blocks, differential LRs (`backbone=1e-6`, `head=5e-5`)
- Augmentation: flip, color jitter, affine, Gaussian blur, MixUp (α=0.8), CutMix (α=1.0)
- Class imbalance: `WeightedRandomSampler` + FocalLoss with label smoothing

**Output:** `MyDrive/DriveGuard/models/vit_spatial_model_v1.pth`

---

### Step 2 — Extract features (`photo_to_tensor_transformers.ipynb`) — [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1q6V7luC6tTFwmoJPi6sYaIUI3kAFAveH?usp=sharing)

**Input:**
- `MyDrive/DriveGuard/models/vit_spatial_model_v1.pth`
- `ds_driveguard_temporal_roi/` uploaded as zip to `MyDrive/DriveGuard/`

- Loads trained spatial model, removes classifier head (`model.reset_classifier(0)`)
- SigLIP normalization: mean=std=0.5, resize to 384×384
- Per sequence: 16 frames → batch → ViT → reshape to `(16, 1152)` → save as float32 .npy

> `DATASET_PATH` in the notebook must match the folder name inside the zip (`ds_driveguard_temporal_roi`), not the zip filename.

**Output:** `MyDrive/DriveGuard/ViT_Features/{train,val,test}/{class}/{seq_id}.npy`

---

### Step 3 — Train temporal model (`temporal_transformer_head.ipynb`) — [![Open in Drive](https://img.shields.io/badge/Open-Google%20Drive-blue)](https://drive.google.com/file/d/1qzFcsufgvXhpvopFnI_FtDi69xdzOTug/view?usp=sharing)

**Input:** `MyDrive/DriveGuard/ViT_Features/{train,val,test}/{class}/a_column_co_driver_*.npy`

- Ablation study across 8 configs (dropout, noise, phone class weight, focal loss)
- Winner selected by Phone F1, tiebroken by macro F1
- Full retrain of winner: Phase 1 head warmup (5 epochs) + Phase 2 differential LRs (up to 40 epochs, early stopping patience=10)
- Architecture: 4-layer Pre-LN Transformer, 8 heads, 768 hidden dim, CLS token

**Output:** `MyDrive/DriveGuard/models/temporal_head_model.pth`

---

### Step 4 — Evaluate end-to-end (`evaluate_pipeline_driveguard.ipynb`)

**Input:**
- `MyDrive/DriveGuard/models/vit_spatial_model_v1.pth`
- `MyDrive/DriveGuard/models/temporal_head_model.pth`
- `ds_driveguard_temporal_roi/` zip

- Runs full two-stage pipeline on test set (no cached features)
- Batch: 16 clips → 256 ViT forward passes, then temporal inference
- AMP: bfloat16 on A100, float16 on V100

**Output:** Classification report (precision/recall/F1 per class), confusion matrix PNG

---

### Step 5 — Run inference (`infer.py`)

Weights and sample video are **auto-downloaded from Google Drive** on first run — no manual setup needed:

```bash
python infer.py
```

Or with custom paths:

```bash
python infer.py \
    --video /path/to/video.mp4 \
    --spatial_weights /path/to/vit_spatial_model_v1.pth \
    --temporal_weights /path/to/temporal_head_model.pth \
    --output_video ./output.mp4
```

Device auto-detected: Apple Silicon MPS (FP16) → CUDA (FP16) → CPU (FP32)

**Real-time performance** (measured on sample video):

| Metric | Value |
|---|---|
| Avg spatial inference | ~11 ms / sampled frame |
| Avg temporal inference | ~2 ms / window |
| Real-time budget | 200 ms / 6 frames |
| GPU target | real-time ✓ |
| CPU (Mac) | ~0.56x real-time (bottleneck: YOLO + I/O, not models) |

**Key parameters:**

| Constant | Value | Purpose |
|---|---|---|
| `WINDOW_FRAMES` | 16 | Temporal window size |
| `STEP` | 6 | Frame sampling stride |
| `CYCLE_SIZE` | 90 | ROI refresh interval |
| `IMG_SIZE` | 384 | ViT input resolution |
| `ROI_PADDING` | 0.08 | YOLO bounding box padding |
| `YOLO_CONF` | 0.25 | Detection confidence threshold |

Frames 0–89: YOLO locks ROI, collects features, shows "Initializing..."
Frame 90+: each new feature triggers temporal inference immediately

---

## Google Drive Layout

```
MyDrive/DriveGuard/
├── models/
│   ├── vit_spatial_model_v1.pth
│   └── temporal_head_model.pth
├── all_cams_ds_driveguard_spatial_roi.zip
├── all_cams_ds_driveguard_temporal_roi.zip
└── ViT_Features/
    ├── train/
    ├── val/
    └── test/
```
