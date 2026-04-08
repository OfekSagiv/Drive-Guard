# Pipeline: Distillation (Swin3D Student)


It trains a **Video Swin3D student model** directly on 16-frame ROI clips and includes:
- Dataset extraction from raw videos and activity CSVs
- Student training
- Test confusion-matrix evaluation
- Random test-sequence prediction visualization

---

## Run Order

### Step 1 - Extract 16-frame ROI dataset

Script: `temporal_extract_all_cams_16frames_roi.py`

```bash
python temporal_extract_all_cams_16frames_roi.py \
  --data_root . \
  --data_dirs data "data 2" "data 3" "data 4" "data 5" "data 6" \
  --output_base ds_driveguard_16frames_roi.nosync
```

Expected structure under `data_root`:
- `activities_3s/<camera>/midlevel.chunks_90.split_0.{train,val,test}.csv`
- Video files inside one or more data folders (`data`, `data 2`, ...)

Output:
- `ds_driveguard_16frames_roi.nosync/{train,val,test}/{Safe,Drink,Phone}/{sequence_id}/frame_00.jpg ... frame_15.jpg`

---

### Step 2 - Train student model

Script: `train_model.py`

```bash
python train_model.py \
  --data_root ds_driveguard_16frames_roi.nosync \
  --output_dir checkpoints \
  --epochs 10 \
  --batch_size 2 \
  --pretrained
```

Outputs:
- `checkpoints/best_swin3d_driveguard.pt`
- `checkpoints/last_swin3d_driveguard.pt`

---

### Step 3 - Evaluate on test set

Script: `evaluate_confusion.py`

```bash
python evaluate_confusion.py \
  --data_root ds_driveguard_16frames_roi.nosync \
  --checkpoint checkpoints/best_swin3d_driveguard.pt
```

Output:
- Printed confusion matrix and accuracy
- Optional plot saved to `checkpoints/confusion_matrix_test.png`

---

### Step 4 - Predict one random test clip

Script: `predict.py`

```bash
python predict.py \
  --data_root ds_driveguard_16frames_roi.nosync \
  --checkpoint checkpoints/best_swin3d_driveguard.pt
```

Output:
- Predicted class + confidence
- 4x4 frame visualization for the sampled sequence
