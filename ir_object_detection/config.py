"""
Shared configuration for the IR object-detection dataset pipeline.

Single source of truth for taxonomy, paths, prompts, and thresholds used by all
four stages (1_sample_ir_frames -> 2_auto_annotate -> 3_review_refine ->
4_build_dataset).

The pipeline is NOT fully autonomous: stages 2-3 are AI-assisted and *propose*
labels; the human-validation gate in stage 4 / README is what accepts them.
"""

import os

# ─── Paths ──────────────────────────────────────────────────────────────────
# DATA_ROOT points at the repo root that holds data/ and activities_3s/.
# Override via env var when running on Colab (e.g. /content/drive/MyDrive/DriveGuard).
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
# Image source and output base are env-overridable so the same pipeline can run on
# an alternate image set without touching defaults:
#   IR_SAMPLED_DIR=/some/other/images  IR_OUT_DIR=/some/out  python 2_auto_annotate.py
# Default is the co-driver ROI crops produced by extract_roi_variant.py.
SAMPLED_DIR = os.environ.get("IR_SAMPLED_DIR") or os.path.join(PKG_DIR, "sampled_frames_roi")
_OUT = os.environ.get("IR_OUT_DIR") or PKG_DIR
MANIFEST_CSV = os.path.join(SAMPLED_DIR, "manifest.csv")
RAW_PRED_DIR = os.path.join(_OUT, "raw_predictions")           # stage 2 output
PREDICTIONS_JSONL = os.path.join(RAW_PRED_DIR, "predictions.jsonl")
REFINED_DIR = os.path.join(_OUT, "refined")                    # stage 3 output
REFINED_LABELS_DIR = os.path.join(REFINED_DIR, "labels")
REVIEW_JSONL = os.path.join(REFINED_DIR, "review.jsonl")
FLAGGED_DIR = os.path.join(REFINED_DIR, "flagged")
DATASET_DIR = os.path.join(_OUT, "dataset")                    # stage 4 output
QC_REPORT_MD = os.path.join(DATASET_DIR, "qc_report.md")
QC_STATS_JSON = os.path.join(DATASET_DIR, "qc_stats.json")

# ─── Taxonomy ───────────────────────────────────────────────────────────────
# Final detector classes (index == YOLO class id).
# bottle + cup are merged into one `drink` container class: the held-out gold set
# contained only bottles (no cups), GroundingDINO splits containers across both
# terms, and `drinking` activities give no reliable bottle/cup distinction.
CLASSES = ["phone", "drink", "food"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# GroundingDINO open-vocabulary prompt. Phrases are period-separated per GD spec.
# Keep both bottle+cup terms so GD fires on either; both map to `drink`.
# NOTE: expanding the food side with precise synonyms (sandwich/banana/bread/apple/
# wrapper/lunch box...) was tested and made food recall WORSE — 9/35 -> 0/35 at every
# box/text threshold. GroundingDINO dilutes per-token text scores on long prompts, so
# the extra terms killed the one token that actually fired ("food"), and IR food blobs
# don't specifically match the precise terms. Keep the single generic "food".
GROUNDING_PROMPT = "cell phone . bottle . cup . food ."

# Map free-text detector/VLM phrases onto our classes. Lower-cased substring match.
PHRASE_TO_CLASS = {
    "cell phone": "phone", "phone": "phone", "smartphone": "phone", "mobile": "phone",
    "bottle": "drink", "water bottle": "drink", "can": "drink", "flask": "drink",
    "cup": "drink", "mug": "drink", "glass": "drink", "tumbler": "drink",
    "food": "food", "snack": "food", "sandwich": "food", "fruit": "food", "banana": "food",
}

# ─── Drive&Act source mapping (mirrors extract_spatial_roi_ds.py) ────────────
CAMERA_MAPPING = {
    "inner_mirror": "ids_1",
    "a_column_co_driver": "ids_2",
    "ceiling": "ids_3",
    "steering_wheel": "ids_4",
    "a_column_driver": "ids_5",
}

# Granular Drive&Act activity -> our coarse activity grouping.
# Object-bearing activities (Phone/Drink) are prioritized in sampling; Safe is
# used only for hard negatives.
PHONE_ACTIVITIES = {"interacting_with_phone", "talking_on_phone"}
DRINK_ACTIVITIES = {"drinking", "opening_bottle", "closing_bottle", "eating", "preparing_food"}
SAFE_ACTIVITIES = {
    "sitting_still", "looking_or_moving_around (e.g. searching)",
    "fastening_seat_belt", "unfastening_seat_belt",
    "putting_on_sunglasses", "taking_off_sunglasses",
}
OBJECT_ACTIVITIES = PHONE_ACTIVITIES | DRINK_ACTIVITIES
ALL_ACTIVITIES = OBJECT_ACTIVITIES | SAFE_ACTIVITIES

SPLITS = ["train", "val", "test"]  # source CSV splits to draw frames from

# ─── Stage 1: class-aware sampling ───────────────────────────────────────────
RANDOM_SEED = 42
PER_CLASS_TARGET = 500        # target frames per intended object class
SAFE_COUNT = 150              # small controlled negative pool (kept only as needed)

# activity -> intended object class (sampling proxy; the TRUE class is set at
# detection in Stage 2/3). cup has no dedicated Drive&Act activity, so `drinking`
# maps to a bottle_or_cup pool and cup's real count is reported after detection.
SAMPLE_OBJ_MAP = {
    "interacting_with_phone": "phone", "talking_on_phone": "phone",
    "opening_bottle": "drink", "closing_bottle": "drink", "drinking": "drink",
    "eating": "food", "preparing_food": "food",
}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}  # participant-grouped

# Frozen dataset spec: the fixed co-driver class-balanced set (a_column_co_driver,
# phone 800 / drink 376 / food 800 / safe 200 = 2176, participant-grouped splits).
# extract_roi_variant.py re-extracts identical ROI crops from this manifest.
FROZEN_MANIFEST = os.path.join(PKG_DIR, "manifest_codriver_roi.csv")

# ─── Stage 2: GroundingDINO + SAM2 ───────────────────────────────────────────
GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"
GD_TEXT_THRESHOLD = 0.25
#
# TWO-LEVEL THRESHOLD STRATEGY. Opus can only review boxes GroundingDINO proposes,
# so Stage 2 OVER-GENERATES for candidate coverage and Stage 3 does the precision.
#
# 1) CANDIDATE thresholds (permissive) — Stage 2 KEEPS every box scoring >= these and
#    preserves its score+phrase in predictions.jsonl. Goal is candidate COVERAGE, not
#    clean labels. Food is faint on IR so its bar is lowest. Stage 2 runs GD post-
#    processing at GD_MIN_CANDIDATE so even the faintest survive, then keeps each box
#    that clears its own class candidate bar. Frames with NO candidate are logged, NOT
#    written as empty YOLO labels (an empty label = "no object" would poison recall).
GD_CANDIDATE_BOX_THRESHOLD = {"phone": 0.20, "drink": 0.20, "food": 0.15}
GD_MIN_CANDIDATE = min(GD_CANDIDATE_BOX_THRESHOLD.values())   # GD post-process bar
#
# 2) ACCEPT thresholds — used by Stage 3 AFTER the spatial/keypoint gate: a gated box
#    scoring >= accept is auto-accepted; a gated box in [candidate, accept) goes to
#    Opus / manual review. (The gate runs first, so high-conf off-driver console FPs
#    are dropped before the accept bar ever applies.) These are the precision-tuned
#    bars from the ROI threshold sweep.
GD_ACCEPT_BOX_THRESHOLD = {"phone": 0.35, "drink": 0.35, "food": 0.22}
#
GD_BOX_THRESHOLD = GD_MIN_CANDIDATE   # legacy fallback name
SAM2_WEIGHTS = "sam2_b.pt"    # via ultralytics.SAM (no extra install)
USE_SAM2 = True               # set False to keep raw GroundingDINO boxes

# ─── Stage 3: review / refinement ────────────────────────────────────────────
# Order: spatial/keypoint gate -> NMS -> per-class accept bands -> Opus -> manual flags.
# Auto-accept vs review is decided PER CLASS by GD_ACCEPT_BOX_THRESHOLD (above): a
# gated box >= its accept bar is auto-accepted; one in [candidate, accept) goes to Opus
# /manual. CONF_HIGH/CONF_LOW remain only as a global fallback for classes without an
# accept entry.
NMS_IOU = 0.6                 # class-wise dedup threshold
CONF_HIGH = 0.45
CONF_LOW = 0.20
# Keypoint gating (reused from pipelines/vit_transformer/infer.py).
KP_CONF = 0.30
RELEVANT_KP = {0, 1, 2, 3, 4, 7, 8, 9, 10}   # nose, eyes, ears, elbows, wrists
OBJ_KP_RADIUS_FRAC = 0.33     # ROI-crop value (from infer.py); kept for reference
# Keypoint-gate radius for Stage 3, for the POINT-TO-BOX gate (_near): a box is kept if a
# relevant keypoint is INSIDE it or within this fraction of the box edge. This replaced the
# old center-to-keypoint test, which dropped large food/wrapper boxes (their center fell
# outside the radius even when overlapping the hand/face). Re-tuned on the balanced-50:
# 0.15 gives 100% food TP retention, ~90% overall TP retention, ~58% FP removal. Smaller
# (0.12) trades food down to ~86% for more FP removal; we favor food retention.
REVIEW_KP_RADIUS_FRAC = 0.15
POSE_WEIGHTS = "yolov8n-pose.pt"
POSE_CONF = 0.25

# VLM review (Anthropic). claude-opus-4-8 is the most capable vision model — best
# for judging small objects in dark IR crops, where label quality drives the
# detector. Override with VLM_MODEL=claude-sonnet-4-6 to cut cost on large runs.
VLM_MODEL = os.environ.get("VLM_MODEL", "claude-opus-4-8")
VLM_CROP_PAD = 0.25           # context padding around the box before sending to VLM
VLM_FLAG_CONF = 0.50          # VLM verdicts below this confidence are flagged for humans

# ─── Stage 4: dataset assembly ───────────────────────────────────────────────
VAL_FRACTION = 0.15           # train/val split, grouped by participant_id
IMG_EXT = ".jpg"
