# DriveGuard

Real-time driver activity recognition using multi-camera vehicle footage.
Classifies driver behavior into three classes: **Safe**, **Drink** (eating/drinking), and **Phone** (phone interaction).

---

## Pipeline Architecture

All pipelines in this repo share the same general structure. What differs between them is the technique used at each stage — and consequently the feature vector dimensions that flow between stages.

```
Raw Video (5 camera angles)
        │
        ▼
┌─────────────────────────────────────────────┐
│           Preprocessing  (shared)           │
│                                             │
│  extract_spatial_roi_ds.py                  │
│    → ds_driveguard_spatial_roi/             │
│      per-frame ROI crops (JPEG)             │
│                                             │
│  extract_temporal_roi_ds.py                 │
│    → ds_driveguard_temporal_roi/            │
│      16-frame sequences (JPEG)              │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│        Spatial Model  (varies per pipeline) │
│                                             │
│  Technique options:                         │
│    CNN, ViT, Knowledge Distillation,        │
│    Pose Skeleton extraction, ...            │
│                                             │
│  Output: feature vectors of shape (D,)      │
│    where D depends on the technique         │
│    e.g. ViT-SO400M → D=1152                 │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│     Feature Extraction  (matches spatial)   │
│                                             │
│  Converts 16-frame sequences into           │
│  .npy tensors of shape (16, D)              │
│  using the trained spatial model as         │
│  a frozen feature extractor                 │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│       Temporal Model  (varies per pipeline) │
│                                             │
│  Technique options:                         │
│    LSTM, Transformer, ...                   │
│                                             │
│  Input: sequence of D-dim feature vectors   │
│  Output: Safe / Drink / Phone               │
└─────────────────────────────────────────────┘
```

---

## Pipelines

| Folder | Spatial Technique | Feature Dim | Temporal Technique | Status |
|---|---|---|---|---|
| `pipelines/vit_transformer` | ViT-SO400M (SigLIP) | 1152 | Transformer encoder | Active |
| `pipelines/cnn_lstm` | EfficientNet-B4 | 1792 | Bidirectional LSTM | Active |
| `pipelines/distillation` | Knowledge Distillation | TBD | TBD | Coming soon |

---

## Preprocessing (Shared)

Both scripts run locally and produce datasets consumed by any pipeline.

### Requirements

```bash
pip install -r requirements.txt
```

> Also download the YOLO pose model:
> ```bash
> python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
> ```

### Step 1 — Spatial dataset

Extracts per-frame ROI crops for spatial model training.

```bash
python extract_spatial_roi_ds.py
```

Output: `ds_driveguard_spatial_roi/{train,val,test}/{Safe,Drink,Phone}/{camera}/`

- Safe: 1 frame per chunk (middle frame)
- Drink / Phone: 3 frames per chunk (start+5, mid, end-5) — minority class boost
- Falls back to full frame if YOLO finds no person

### Step 2 — Temporal dataset

Extracts 16-frame sequences for temporal model training.

```bash
python extract_temporal_roi_ds.py
```

Output: `ds_driveguard_temporal_roi/{train,val,test}/{Safe,Drink,Phone}/{seq_id}/frame_00.jpg … frame_15.jpg`

- Balanced per camera by undersampling to median class count
- 16 frames sampled evenly across each activity chunk via `np.linspace`

### Data structure required

```
data/
  {camera_name}/
    {vp_folder}/
      {file_id}.mp4

activities_3s/
  {camera_name}/
    midlevel.chunks_90.split_0.train.csv
    midlevel.chunks_90.split_0.val.csv
    midlevel.chunks_90.split_0.test.csv
```

Camera names: `inner_mirror`, `a_column_co_driver`, `ceiling`, `steering_wheel`, `a_column_driver`

### Activity → class mapping

| Raw activity label | Class |
|---|---|
| sitting_still, looking_or_moving_around, fastening/unfastening_seat_belt, putting_on/taking_off_sunglasses | Safe |
| drinking, opening/closing_bottle, eating, preparing_food | Drink |
| interacting_with_phone, talking_on_phone | Phone |

---

## Quick Start

```bash
git clone https://github.com/OfekSagiv/Drive-Guard.git
cd Drive-Guard
pip install -r requirements.txt

# Run preprocessing
python extract_spatial_roi_ds.py
python extract_temporal_roi_ds.py

# Then follow the README inside the pipeline of your choice
cd pipelines/vit_transformer
```
