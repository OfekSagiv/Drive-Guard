# Pipeline: Distillation (Swin3D Student)


It trains a **Video Swin3D student model** directly on 16-frame ROI clips and includes:
- Dataset extraction from raw videos and activity CSVs
- Student training
- Test confusion-matrix evaluation
- Random test-sequence prediction visualization

---

## Run Order

### Prerequisites
- Raw videos + `activities_3s/<camera>/midlevel.chunks_90.split_0.{train,val,test}.csv` available locally
- Runs locally (CPU/MPS/CUDA auto-detected) — no Colab/Google Drive dependency

---

### Step 1 - Extract 16-frame ROI dataset (`temporal_extract_all_cams_16frames_roi.ipynb`)

Edit the config cell (`DATA_ROOT`, `DATA_DIR_NAMES`, `OUTPUT_BASE`), then run all cells.

Expected structure under `DATA_ROOT`:
- `activities_3s/<camera>/midlevel.chunks_90.split_0.{train,val,test}.csv`
- Video files inside one or more data folders (`data`, `data 2`, ...)

Output:
- `ds_driveguard_16frames_roi.nosync/{train,val,test}/{Safe,Drink,Phone}/{sequence_id}/frame_00.jpg ... frame_15.jpg`

---

### Step 2 - Train student model (`train_model.ipynb`)

Edit the `cfg` dict (`data_root`, `output_dir`, `epochs`, `batch_size`, `pretrained`, optional `enable_distillation` + `teacher_checkpoint`), then run all cells.

Outputs:
- `checkpoints/best_swin3d_driveguard.pt`
- `checkpoints/last_swin3d_driveguard.pt`

---

### Step 3 - Evaluate on test set (`evaluate_confusion.ipynb`)

Edit the `cfg` dict (`data_root`, `checkpoint`), then run all cells.

Output:
- Printed confusion matrix and accuracy
- Optional plot saved to `checkpoints/confusion_matrix_test.png`

---

### Step 4 - Predict one random test clip (`predict.ipynb` / `infer.py`)

Notebook: edit the `cfg` dict (`data_root`, `checkpoint`), then run all cells.

Or run locally as a script:

```bash
python infer.py \
  --data_root ds_driveguard_16frames_roi.nosync \
  --checkpoint checkpoints/best_swin3d_driveguard.pt
```

Output:
- Predicted class + confidence
- 4x4 frame visualization for the sampled sequence
