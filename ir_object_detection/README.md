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

The COCO fusion experiment result that established the technical trigger: `yolov8m.pt` achieved only **~24% in-ROI clip detection rate** at mean confidence ~0.46 on Drive&Act IR, producing **W\* = 0.0** (optimizer turned fusion off) and **McNemar p = 1.0** — zero statistically significant improvement over the temporal baseline at 0.9399 accuracy. See `docs/fusion_experiment.md` for the full experiment log.

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

### Stage 1 — class-aware sampling
Reads the Drive&Act activity CSVs and takes the **middle frame** of each labeled
chunk (`(frame_start+frame_end)//2`). Sampling is **class-aware** (not the natural
activity distribution): it targets ~`PER_CLASS_TARGET` (500) frames per intended
object class via an activity→class proxy, plus a small controlled Safe-negative
pool. Frames are kept at **full resolution** (objects are small, and live inference
detects on the full frame). It bakes a participant-grouped **train/val/test**
split into the manifest and prints a per-class **ceiling report**.

> **Object class ≠ activity.** Drive&Act annotates activities, not objects:
> `phone←interacting/talking_on_phone`, `bottle←opening/closing_bottle`,
> `food←eating/preparing_food`. **`cup` has no dedicated activity** — it only
> appears inside `drinking` frames (a `bottle_or_cup` pool), so the whole drinking
> pool is taken and cup's real count is **reported after detection** (Stage 3),
> not guaranteed at sampling. See `SAMPLE_OBJ_MAP` in `config.py`.

### Stage 2 — auto-annotate (GroundingDINO + SAM2)
- **GroundingDINO** (`IDEA-Research/grounding-dino-base`), an open-vocabulary
  detector prompted with text (`"cell phone . bottle . cup . food ."`) — far more
  robust on IR than a COCO-fixed model. Proposes boxes + scores.
- **SAM2** (`sam2_b.pt`, via `ultralytics`) tightens each box from its mask.
- Phrases are mapped to the 4 classes; **confidence is preserved** in
  `predictions.jsonl` (YOLO `.txt` can't hold scores).

Threshold calibration history (explains the values in `config.py`):
- GroundingDINO at default `0.12/0.12` over-detected massively (~10 boxes/img on ROI crops). Sweep settled on `box=0.35, text=0.25`.
- **Food = 0 detections at any threshold** on raw IR. Food prompt expansion (23 precise terms: `sandwich . snack . banana . bread . apple . wrapper …`) was tried and **failed** — long prompts dilute per-token GD text scores, killing the one term (`food`) that had any recall. The current `"cell phone . bottle . cup . food ."` is the result of that failure; don't try to expand it.
- Per-class thresholds (`phone/drink: 0.35, food: 0.22`) were adopted after food recall improved from 1/35 → 9/35 at the lower bar with zero added FPs on non-food images.
- CLAHE + gamma enhancement (`--enhance` flag): helps drink +30% and food +11% detection recall, but adds console FPs — pair with Stage-3 cleanup. Tested for Opus verification crops too: **0 recovery, adds noise** — enhancement belongs at detection, not verification.

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

**Annotation approach evolution — why `build_rgb_transfer.py` exists:**

The dark-IR pipeline above hit a ceiling: food recall capped at ~26–46% and console FPs on drinking frames could not be resolved by threshold tuning (console detections fire at *higher* confidence than real drinking-at-face boxes, so raising the threshold kills TPs first). The breakthrough was to stop fighting IR detection and instead leverage the Drive&Act co-driver camera's synchronized RGB/IR pair:

| Approach | Food recall | Phone recall | Drink recall |
|---|---|---|---|
| Dark-IR path (GD+SAM2, Stages 2→3) | ~26–46% | better, but console FPs | better, but console FPs |
| **RGB→IR transfer (`build_rgb_transfer.py`, Path 1d)** | **~63%** | **~78%** | **~91%** |

Transfer method: detect on bright RGB (`kinect_color`) with GroundingDINO → filter boxes near wrist keypoints (drops console/lap FPs) → warp into IR via per-frame homography fit from 3D-projected body joints (RGB joints from YOLO-pose; IR joints from `openpose_3d` + camera calibration). Full-scale on 2,176 frames: **1,517 labeled frames (≥1 box)**, 2,210 total boxes (phone 854 · drink 1,017 · food 339), 157 Safe negatives. Full experiment log and all negative results (failed prompt expansion, failed homography variants, CLAHE-on-Opus failure): `docs/ir_dataset_experiments.md`.

**Parallel live probe — YOLO-World on the Reolink IR camera:** `pipelines/vit_transformer/explore_yoloworld_ir.py` is testing YOLO-World as a one-directional Phone verifier on the higher-quality Reolink IR stream (much better contrast than Drive&Act 2014 NIR). Early results show `smartphone` detecting at 0.57–0.88 conf when held and **0 detections on empty-hand frames** — a key property the COCO detector never had. If a labeled driving-position test (phone-to-ear vs hand-to-ear) confirms clean separation, a verifier that *demotes* Phone→Safe when no phone is detected can be integrated without waiting for the full YOLO fine-tuning pipeline.

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

## Fixed class-balanced dataset (frozen)

`manifest_balanced.csv` is a **committed, frozen dataset spec**: a fixed
class-balanced set (~500 frames each for phone / bottle / food, the full
`drinking` pool for bottle/cup, + a small Safe-negative set), with a
participant-grouped **train/val/test** split baked in. The manifest is the source
of truth; the images are large and regenerated from it + the raw videos on demand.
Always work on this same set:

```bash
# reproduce the exact frames (identical every time, any machine):
python 1_sample_ir_frames.py --frozen manifest_balanced.csv
python 2_auto_annotate.py
python 3_review_refine.py
python 4_build_dataset.py
```

Selection is deterministic (regenerates byte-identically); the manifest is
committed so the set never drifts even if the sampling logic changes. To define a
*new* frozen set: `--per-class N --safe M --no-extract --freeze-to <new.csv>`,
then commit `<new.csv>`.

## Run order (ad-hoc sampling)

```bash
# preview first (small/fast/free) to tune config.py thresholds
python 1_sample_ir_frames.py --per-class 8 --safe 4
python 2_auto_annotate.py --limit 10
python 3_review_refine.py --limit 10 --no-vlm

# full run (Opus review on, all frames; --per-class defaults to 500)
python 1_sample_ir_frames.py
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

### Building the gold set + measuring — `make_gold_set.py` / `evaluate_labels.py`

The gold set is a small set of images **you hand-label** as ground truth.

1. **Create it** with `make_gold_set.py` — a leakage-free stratified sample
   (default 150) drawn ONLY from the held-out `val`/`test` splits, so it never
   overlaps train participants. It extracts the frames into `gold/images/`, writes
   **empty** label stubs into `gold/labels/`, and a `gold_manifest.csv` tracking
   each image (activity, camera, participant, split, stratum):
   ```bash
   python make_gold_set.py --n 150            # stratified: phone / food / bottle_cup_pool / safe
   ```
2. **Label each image by hand** in `gold/labels/` (YOLO format,
   `class cx cy w h`, classes `phone=0 bottle=1 cup=2 food=3`); an image with no
   target objects stays an empty `.txt`. Be sure to **add** boxes for real objects
   — labeling from scratch (empty stubs) keeps recall honest.
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

> Do the gold measurement on the frozen set's **val/test split**, not the preview
> frames, for a representative training decision. Pay special attention to **cup**
> recall — it's the class with no source activity and the lowest ceiling.

`dataset/qc_report.md` summarizes class balance, negatives, decision mix, box
sizes, and the participant split to inform the audit.

## Files

| File | Role |
|---|---|
| `config.py` | taxonomy, class-aware sampling map, thresholds, prompts, `VLM_MODEL` |
| `manifest_balanced.csv` | committed frozen dataset spec (class-balanced, train/val/test) |
| `1_sample_ir_frames.py` … `4_build_dataset.py` | the 4 pipeline stages |
| `make_gold_set.py` | build a leakage-free stratified gold set (held-out splits) |
| `evaluate_labels.py` | precision/recall vs gold set (the training gate) |
| `visualize_labels.py` | render boxes for visual inspection |
| `build_ir_detection_ds.ipynb` | Colab GPU wrapper (stages 1→4) |
| `requirements.txt` | extra deps (transformers, accelerate, anthropic, fiftyone) |
