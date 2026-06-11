# IR Object-Detection Dataset Builder

Builds a high-quality **infrared** (NIR) object-detection dataset from the shared
Drive&Act preprocessing pipeline, so a YOLO detector can later be fine-tuned to
find **phones, bottles, cups, and food** in the dark in-car footage.

## Why

The `vit_transformer` pipeline already tried fusing object detections into the
temporal classifier and it didn't help — the COCO-pretrained `yolov8m.pt` is
out-of-domain on IR cabin footage (only ~29.5% detection rate, zero fusion
benefit, best fusion weight W=0; see `pipelines/vit_transformer/eval_infer_full.log`).
The fix is an **in-domain detector**, which needs an IR detection dataset — this
folder produces it.

## Pipeline (4 stages)

```
1_sample_ir_frames.py  → sampled_frames/*.jpg + manifest.csv
2_auto_annotate.py     → raw_predictions/predictions.jsonl + raw YOLO labels   (GroundingDINO + SAM2)
3_review_refine.py     → refined/labels/*.txt + review.jsonl + flagged/         (heuristics + Claude vision)
4_build_dataset.py     → dataset/{images,labels}/{train,val} + data.yaml + qc_report.md
```

This is **not fully autonomous**: stages 2–3 are AI-assisted and *propose*
labels; the human-validation gate below accepts them before any training.

## Setup

```bash
source ../.venv/bin/activate
pip install -r requirements.txt          # transformers, accelerate, anthropic, (fiftyone)
export ANTHROPIC_API_KEY=sk-...          # for the Stage 3 VLM pass (optional; without it, uncertain dets are flagged)
```

GroundingDINO and SAM2 effectively need a GPU — run the full pass on **Colab GPU**
(see `build_ir_detection_ds.ipynb`). Local MPS works for small previews.

## Run order

```bash
python 1_sample_ir_frames.py --target 500          # ~500 middle frames, Phone/Drink prioritized + Safe negatives
python 2_auto_annotate.py                          # GroundingDINO + SAM2 → boxes + confidence
python 3_review_refine.py                           # NMS + keypoint gate + VLM review  (add --no-vlm to skip Claude)
python 4_build_dataset.py                           # final YOLO dataset + QC report
```

Preview flags: `2_auto_annotate.py --limit 10`, `3_review_refine.py --limit 10` —
tune `config.py` thresholds on a handful of frames first.

All knobs (taxonomy, thresholds, prompts, paths, VLM model) live in `config.py`.

## Key design choices

- **Middle frame** per chunk (reuses `(start+end)//2` from `extract_spatial_roi_ds.py`).
- **Full-resolution frames**, not person-ROI crops — objects are small and live
  inference runs the detector on the full frame.
- **Keypoint gating** reuses the hand/face proximity filter from
  `pipelines/vit_transformer/infer.py` to drop objects not near the driver.
- **Confidence preserved** in `predictions.jsonl` / `review.jsonl` (YOLO `.txt`
  can't hold scores).
- **Train/val split grouped by `participant_id`** — no person spans both sets.

## Human-validation gate (REQUIRED before training)

The auto-pipeline output is a *proposal*. Before fine-tuning a detector on it:

1. **Review 100% of `refined/flagged/`** (the uncertain band).
2. **Audit a random 10%** of auto-accepted labels.
3. Build a **~50-image gold set**, hand-labeled independently; measure pipeline
   precision/recall against it.
4. **Acceptance gate:** proceed only if gold-set **precision ≥ 0.90** and
   **recall ≥ 0.80**; otherwise re-tune Stage 2/3 thresholds and re-run.
5. Suggested visual-audit tool: **FiftyOne** (`pip install fiftyone`, native YOLO
   + bbox editing); Label Studio / CVAT are alternatives.

`dataset/qc_report.md` summarizes class balance, negatives, decision mix, box
sizes, and the participant split to inform the audit.
