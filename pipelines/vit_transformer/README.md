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

### Step 1 — Train spatial model (`ViT_spatial_model.ipynb`) — [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mJ1FgFPpd1Xi1gEtz_d-yVYDShJ9St2S)

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

### Step 3 — Train temporal model (`temporal_transformer_head.ipynb`) — [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qzFcsufgvXhpvopFnI_FtDi69xdzOTug)

**Input:** `MyDrive/DriveGuard/ViT_Features/{train,val,test}/{class}/a_column_co_driver_*.npy`

- Ablation study across 8 configs (dropout, noise, phone class weight, focal loss)
- Winner selected by Phone F1, tiebroken by macro F1
- Full retrain of winner: Phase 1 head warmup (5 epochs) + Phase 2 differential LRs (up to 40 epochs, early stopping patience=10)
- Architecture: 4-layer Pre-LN Transformer, 8 heads, 768 hidden dim, CLS token

**Output:** `MyDrive/DriveGuard/models/temporal_head_model.pth`

---

### Step 4 — Evaluate end-to-end (`evaluate_pipeline_driveguard.ipynb`) — [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xpjUvB4hrzQgttQW9XVTSg0EcmxJfqNX)

Runs the **full two-stage pipeline** on the test set with **no cached `.npy` features** — the ViT
encodes the raw JPEG frames live, exactly as `infer.py` does. This is the honest end-to-end number;
Step 3's test metrics are computed from pre-extracted features and so do not exercise the spatial
model at evaluation time.

**Input** (all paths as hardcoded in Cell 2 — these are *not* under `models/`):
- `MyDrive/DriveGuard/best_model_fused_internViT.pth` — spatial weights (1.6 GB)
- `MyDrive/DriveGuard/stage4_single_cam/best_stage4_single_cam_model.pth` — temporal weights (~53 MB)
- `MyDrive/DriveGuard/stage4_single_cam/cfg_stage4_single_cam.json` — temporal architecture cfg,
  loaded so the model definition always matches the checkpoint
- `MyDrive/DriveGuard/ds_driveguard_16frames_roi.nosync.zip` — 16-frame clip dataset (~28 GB;
  copied to local disk and extracted, since Drive FUSE is slow for files this size)

**Run details:**
- Filters to a single camera (`CFG['camera'] = 'a_column_co_driver'`) → 782 test sequences
  (311 Drink / 148 Phone / 323 Safe), 49 batches
- Batch: 16 clips → 256 ViT forward passes per step, then temporal inference (drop to 8 if OOM)
- Spatial 428.2 M params + temporal 13.2 M params, both under `torch.compile`
- AMP: bfloat16 on A100, float16 on V100; `noise_std` forced to 0.0 at eval (training used 0.075)

**Output:** classification report + confusion matrix PNG, written to
`MyDrive/DriveGuard/eval_pipeline/` (`classification_report_e2e.txt`, `confusion_matrix_e2e.png`).

**Measured result** (A100-40GB run):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Drink | 0.99 | 0.93 | 0.96 | 311 |
| Phone | 0.88 | 0.86 | 0.87 | 148 |
| Safe | 0.93 | 0.99 | 0.96 | 323 |
| **Accuracy** | | | **0.9425** | 782 |
| **Macro F1** | | | **0.9301** | 782 |

Overall accuracy and Macro F1 land within a point of the cached-feature evaluation in Step 3
(0.94 / 0.93), but the per-class balance shifts: end-to-end trades Phone precision (0.98 → 0.88)
for Phone recall (0.82 → 0.86), and improves Safe precision (0.88 → 0.93).

---

### Step 5 — Run inference (`infer.py` / `vit_transformer_head_inference.ipynb`) — [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qTJOw8dXBFVhdIXcgqWVw0U4j1Tf1IPX)

The Colab notebook (`vit_transformer_head_inference.ipynb`) runs the full two-stage pipeline on a single video in Colab — loads both model weights from Drive, detects the person ROI with YOLO, extracts ViT spatial features, and classifies behavior with the temporal Transformer head. Use it for quick GPU-accelerated testing without a local environment.

Alternatively, run locally with `infer.py`. Weights and sample video are **auto-downloaded from Google Drive** on first run — no manual setup needed:

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

### Step 6 — Evaluate the inference / object-detection fusion (`evaluate_infer.py`)

Quantifies whether the object-detection fusion in `infer.py` actually helps, by
running `infer.py`'s own functions over the labeled local test set and comparing
**baseline (W=0)** vs **fused** predictions. Tunes the fusion weight `W` on the
**val** split, locks `W*`, then reports the A/B on **test** (per-class P/R/F1,
confusion matrices, exact McNemar). Runs locally — no Colab.

```bash
python evaluate_infer.py                 # full a_column_co_driver eval
python evaluate_infer.py --limit 5       # fast smoke run
python evaluate_infer.py --no_kp_gate    # diagnose keypoint-gate suppression
```

Caveats (printed at runtime): the object detector runs on the pre-cropped 384px
ROI (not the full frame as in production), and live ROI-locking / `STEP` sampling
are bypassed — so this measures the fusion **decision math** on labeled windows,
not a full streaming replay. Defaults to `--camera a_column_co_driver` (the only
camera the temporal model was trained on).

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
