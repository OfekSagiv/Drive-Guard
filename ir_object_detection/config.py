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
SAMPLED_DIR = os.path.join(PKG_DIR, "sampled_frames")          # stage 1 output
MANIFEST_CSV = os.path.join(SAMPLED_DIR, "manifest.csv")
RAW_PRED_DIR = os.path.join(PKG_DIR, "raw_predictions")         # stage 2 output
PREDICTIONS_JSONL = os.path.join(RAW_PRED_DIR, "predictions.jsonl")
REFINED_DIR = os.path.join(PKG_DIR, "refined")                 # stage 3 output
REFINED_LABELS_DIR = os.path.join(REFINED_DIR, "labels")
REVIEW_JSONL = os.path.join(REFINED_DIR, "review.jsonl")
FLAGGED_DIR = os.path.join(REFINED_DIR, "flagged")
DATASET_DIR = os.path.join(PKG_DIR, "dataset")                 # stage 4 output
QC_REPORT_MD = os.path.join(DATASET_DIR, "qc_report.md")
QC_STATS_JSON = os.path.join(DATASET_DIR, "qc_stats.json")

# ─── Taxonomy ───────────────────────────────────────────────────────────────
# Final detector classes (index == YOLO class id).
CLASSES = ["phone", "bottle", "cup", "food"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# GroundingDINO open-vocabulary prompt. Phrases are period-separated per GD spec.
GROUNDING_PROMPT = "cell phone . bottle . cup . food ."

# Map free-text detector/VLM phrases onto our 4 classes. Lower-cased substring match.
PHRASE_TO_CLASS = {
    "cell phone": "phone", "phone": "phone", "smartphone": "phone", "mobile": "phone",
    "bottle": "bottle", "water bottle": "bottle", "can": "bottle", "flask": "bottle",
    "cup": "cup", "mug": "cup", "glass": "cup", "tumbler": "cup",
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

# ─── Stage 1: sampling ───────────────────────────────────────────────────────
TARGET_FRAMES = 500           # approximate total images to sample
SAFE_NEGATIVE_FRAC = 0.12     # fraction of TARGET_FRAMES reserved for Safe negatives
RANDOM_SEED = 42

# ─── Stage 2: GroundingDINO + SAM2 ───────────────────────────────────────────
GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"
GD_BOX_THRESHOLD = 0.25
GD_TEXT_THRESHOLD = 0.25
SAM2_WEIGHTS = "sam2_b.pt"    # via ultralytics.SAM (no extra install)
USE_SAM2 = True               # set False to keep raw GroundingDINO boxes

# ─── Stage 3: review / refinement ────────────────────────────────────────────
NMS_IOU = 0.6                 # class-wise dedup threshold
# Confidence bands on GroundingDINO score: >=HIGH auto-accept, <LOW drop, between -> VLM.
CONF_HIGH = 0.45
CONF_LOW = 0.20
# Keypoint gating (reused from pipelines/vit_transformer/infer.py).
KP_CONF = 0.30
RELEVANT_KP = {0, 1, 2, 3, 4, 7, 8, 9, 10}   # nose, eyes, ears, elbows, wrists
OBJ_KP_RADIUS_FRAC = 0.33     # object center must be within frac*frame-size of a keypoint
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
