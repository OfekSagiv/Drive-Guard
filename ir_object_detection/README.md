# IR Object-Detection Dataset Builder

Builds a high-quality **infrared** (NIR) object-detection dataset from the shared
Drive&Act preprocessing pipeline, so a YOLO detector can later be fine-tuned to
find **phone, bottle, cup, food** in the dark in-car footage.

## Why

The `vit_transformer` pipeline already tried fusing object detections into the
temporal classifier and it didn't help — the COCO-pretrained `yolov8m.pt` is
out-of-domain on IR cabin footage (only ~29.5% detection rate, zero fusion
benefit, best fusion weight W=0; see `pipelines/vit_transformer/`).
The fix is an **in-domain detector**, which needs an IR detection dataset — this
folder produces it.

## How it works (4 stages + 2 tools)

```
1_sample_ir_frames.py  → sampled_frames/*.jpg + manifest.csv
2_auto_annotate.py     → raw_predictions/predictions.jsonl + raw YOLO labels   (GroundingDINO + SAM2)
3_review_refine.py     → refined/labels/*.txt + review.jsonl + flagged/         (heuristics + Claude Opus)
4_build_dataset.py     → dataset/{images,labels}/{train,val} + data.yaml + qc_report.md

evaluate_labels.py     → precision/recall of the auto-labels vs a hand-labeled gold set
visualize_labels.py    → renders boxes onto frames into viz/ for visual inspection
```

This is **not fully autonomous**: stages 2–3 are AI-assisted and *propose* labels;
the human-validation gate accepts them before any training.

### Stage 1 — sample frames
Reads the Drive&Act activity CSVs and takes the **middle frame** of each labeled
chunk (`(frame_start+frame_end)//2`). Prioritizes Phone/Drink activities (so
objects are present) and adds ~12% Safe frames as **hard negatives**. Frames are
kept at **full resolution** (objects are small, and live inference detects on the
full frame). Writes `manifest.csv` (camera, participant_id, activity, …).

### Stage 2 — auto-annotate (GroundingDINO + SAM2)
- **GroundingDINO** (`IDEA-Research/grounding-dino-base`), an open-vocabulary
  detector prompted with text (`"cell phone . bottle . cup . food ."`) — far more
  robust on IR than a COCO-fixed model. Proposes boxes + scores.
- **SAM2** (`sam2_b.pt`, via `ultralytics`) tightens each box from its mask.
- Phrases are mapped to the 4 classes; **confidence is preserved** in
  `predictions.jsonl` (YOLO `.txt` can't hold scores).

### Stage 3 — AI-assisted review (heuristics + Claude Opus 4.8)
Two passes that *propose* refinements:

**A. Heuristic pass (free, deterministic)**
- **NMS** merges duplicate detections per class.
- **Keypoint gating** (reuses the hand/face proximity filter from
  `pipelines/vit_transformer/infer.py`) drops detections not near the driver;
  skipped when no person is detected rather than dropping everything.
- **Confidence bands**: `>= CONF_HIGH` → auto-accept; `< CONF_LOW` → drop;
  in-between → send to the VLM.

**B. VLM pass (Claude Opus 4.8 vision)**
- Each uncertain detection is cropped (+context) and sent to Claude, which returns
  a structured verdict: category (`phone|bottle|cup|food|none`) + confidence +
  reason.
- The verdict **drops** false positives (`none`), **relabels** mislabeled boxes,
  and **flags** low-confidence cases for a human.
- Outputs: `refined/labels/*.txt` (clean labels), `refined/review.jsonl` (every
  decision + VLM reason), `refined/flagged/` (crops needing human eyes).

The decision mix (`accept_high`, `accept_vlm`, `drop_vlm_fp`, `drop_keypoint_gate`,
`flag`) is printed at the end and recorded in `qc_report.md`.

### Stage 4 — build dataset + QC
Assembles the Ultralytics YOLO layout (`images/`, `labels/`, `data.yaml`), splits
**train/val by `participant_id`** (no person spans both sets — prevents leakage),
and writes `qc_report.md` + `qc_stats.json` (class balance, negatives, decision
mix, box sizes, participant split).

## Setup

```bash
source ../.venv/bin/activate
pip install -r requirements.txt          # transformers, accelerate, anthropic, (fiftyone)
```

GroundingDINO and SAM2 effectively need a GPU — run the full pass on **Colab GPU**
(see `build_ir_detection_ds.ipynb`). Local Apple-Silicon MPS works for previews.

### Anthropic API key (Stage 3 Opus review)

The VLM review uses **Claude Opus 4.8** (`config.VLM_MODEL`, the most capable
vision model — best for judging small objects in dark IR). Inject the key in the
shell that runs Stage 3:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

- **`export` only affects the current shell** — set it in the same terminal you
  run `3_review_refine.py` from (a new tab won't have it).
- **Verify it's active:** Stage 3 must NOT print `ANTHROPIC_API_KEY not set —
  skipping VLM pass`, and the summary should show non-zero `accept_vlm`/`drop_vlm`.
- **Without a key** (or with `--no-vlm`): the VLM pass is skipped and every
  uncertain detection is **flagged** for manual review instead — the pipeline
  still runs, just with more to review by hand.
- **Cost:** one vision call per uncertain detection — a few hundred small calls
  for ~500 frames (low single-digit dollars on Opus).
- **Cheaper model:** override per-run without editing code:
  `VLM_MODEL=claude-sonnet-4-6 python 3_review_refine.py`

## Run order

```bash
# preview first (small/fast/free) to tune config.py thresholds
python 1_sample_ir_frames.py --target 40
python 2_auto_annotate.py --limit 10
python 3_review_refine.py --limit 10 --no-vlm

# full run (Opus review on, all frames)
python 1_sample_ir_frames.py --target 500
python 2_auto_annotate.py
python 3_review_refine.py
python 4_build_dataset.py
cat dataset/qc_report.md
```

All knobs (taxonomy, thresholds, prompts, paths, VLM model) live in `config.py`.

> Re-run cleanly: stages 1 and 3 do not wipe their own output dirs, so before a
> fresh full run delete the generated dirs to avoid mixing stale artifacts:
> `rm -rf sampled_frames raw_predictions refined dataset viz`

## Inspecting the result — `visualize_labels.py`

Renders boxes onto the frames (phone=red, bottle=green, cup=blue, food=yellow)
into `viz/`, with contact-sheet overviews:

```bash
python visualize_labels.py                # accepted labels  -> viz/accepted/ + _accepted_contact.jpg
python visualize_labels.py --dropped      # also rejected dets -> viz/dropped/ (tagged with reason)
python visualize_labels.py --labels gold/labels   # visualize any YOLO label dir
```

## Human-validation gate (REQUIRED before training)

The auto-pipeline output is a *proposal*. Before fine-tuning a detector:

1. **Review 100% of `refined/flagged/`** (the uncertain band).
2. **Audit a random 10%** of auto-accepted labels.
3. Build a **~50-image gold set** and measure (below).
4. **Acceptance gate:** proceed only if gold **precision ≥ 0.90** and
   **recall ≥ 0.80**; otherwise re-tune Stage 2/3 thresholds and re-run.

### Building the gold set + measuring — `evaluate_labels.py`

The gold set is a small set of images **you hand-label** as ground truth.

1. **Seed it** from the current frames (start from auto-labels to save time):
   copy the images into `gold/images/` and the predicted labels into
   `gold/labels/` (an image with no objects = an empty `.txt`).
2. **Correct each label by hand** in `gold/labels/` (YOLO format,
   `class cx cy w h`, classes `phone=0 bottle=1 cup=2 food=3`):
   - **delete** wrong boxes (keeps precision honest),
   - **fix** loose/mislabeled boxes,
   - **add** boxes for real objects the pipeline missed ← critical, or recall
     looks artificially perfect.
   Tools: **labelImg** (YOLO mode, edits in place), **FiftyOne**, CVAT, Roboflow.
3. **Measure** — greedy IoU matching of predictions against your gold:
   ```bash
   python evaluate_labels.py \
       --gold   gold/labels \
       --pred   refined/labels \
       --images gold/images \
       --iou 0.5
   ```
   Prints per-class + overall **precision/recall** and a `PASS`/`FAIL` line for
   the gate. If it fails, it names the lever: low recall → lower `GD_BOX_THRESHOLD`
   in `config.py` (propose more in Stage 2); low precision → tighten the review.

> Do the gold measurement on the **full 500-frame run's val split**, not the
> preview frames, for a representative training decision.

`dataset/qc_report.md` summarizes class balance, negatives, decision mix, box
sizes, and the participant split to inform the audit.

## Files

| File | Role |
|---|---|
| `config.py` | taxonomy, thresholds, prompts, paths, `VLM_MODEL` |
| `1_sample_ir_frames.py` … `4_build_dataset.py` | the 4 pipeline stages |
| `evaluate_labels.py` | precision/recall vs gold set (the training gate) |
| `visualize_labels.py` | render boxes for visual inspection |
| `build_ir_detection_ds.ipynb` | Colab GPU wrapper (stages 1→4) |
| `requirements.txt` | extra deps (transformers, accelerate, anthropic, fiftyone) |
