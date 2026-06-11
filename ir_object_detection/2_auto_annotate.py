"""
Stage 2 — Auto-annotate sampled IR frames with GroundingDINO (+ optional SAM2).

* GroundingDINO (open-vocabulary) proposes boxes for the prompt
  "cell phone . bottle . cup . food ." — far more domain-flexible than COCO YOLO
  on dark IR cabin footage.
* SAM2 (via ultralytics) optionally tightens each box from its segmentation mask.
* Predicted phrases are mapped onto the 4-class taxonomy (config.PHRASE_TO_CLASS).

Outputs (one detection per line):
  raw_predictions/predictions.jsonl  — image, class, xyxy (px), gd_conf, gd_phrase, sam_iou
Also writes raw YOLO labels per image to raw_predictions/labels/*.txt for preview.

Run: python ir_object_detection/2_auto_annotate.py [--limit N]
"""

import argparse
import json
import os
import sys

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _phrase_to_class(phrase: str):
    """Longest-substring match of a GroundingDINO phrase to a taxonomy class."""
    p = phrase.lower().strip()
    best, best_len = None, 0
    for key, cls in C.PHRASE_TO_CLASS.items():
        if key in p and len(key) > best_len:
            best, best_len = cls, len(key)
    return best


def _load_grounding():
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(C.GROUNDING_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(C.GROUNDING_MODEL).to(DEVICE)
    model.eval()
    return proc, model


def _ground_image(proc, model, rgb):
    """Return list of (xyxy, score, phrase) for one RGB image (H,W,3)."""
    from PIL import Image
    pil = Image.fromarray(rgb)
    inputs = proc(images=pil, text=C.GROUNDING_PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=C.GD_BOX_THRESHOLD, text_threshold=C.GD_TEXT_THRESHOLD,
        target_sizes=[pil.size[::-1]],
    )[0]
    # transformers >=4.51 migrates res["labels"] to integer ids; "text_labels"
    # holds the phrase strings _phrase_to_class needs. Prefer it when present.
    labels = res["text_labels"] if "text_labels" in res else res["labels"]
    out = []
    for box, score, label in zip(res["boxes"], res["scores"], labels):
        out.append((box.tolist(), float(score), str(label)))
    return out


def _refine_with_sam(sam, bgr, box):
    """Return (tightened_xyxy, iou_estimate) from SAM2 mask, or (box, None)."""
    try:
        r = sam(bgr, bboxes=[box], verbose=False)[0]
        if r.masks is None or len(r.masks.data) == 0:
            return box, None
        m = r.masks.data[0].cpu().numpy() > 0.5
        ys, xs = m.nonzero()
        if len(xs) == 0:
            return box, None
        tight = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        return tight, float(m.sum()) / max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    except Exception:
        return box, None


def _to_yolo(xyxy, w, h):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="annotate only first N images (preview)")
    ap.add_argument("--no-sam", action="store_true", help="skip SAM2 box refinement")
    args = ap.parse_args()

    images = sorted(f for f in os.listdir(C.SAMPLED_DIR) if f.endswith(C.IMG_EXT))
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images in {C.SAMPLED_DIR}; run stage 1 first.")

    print(f"Device: {DEVICE} | images: {len(images)}")
    proc, model = _load_grounding()
    use_sam = C.USE_SAM2 and not args.no_sam
    sam = None
    if use_sam:
        from ultralytics import SAM
        sam = SAM(C.SAM2_WEIGHTS)

    os.makedirs(C.RAW_PRED_DIR, exist_ok=True)
    labels_dir = os.path.join(C.RAW_PRED_DIR, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    n_det = 0
    with open(C.PREDICTIONS_JSONL, "w") as jf:
        for i, name in enumerate(images):
            bgr = cv2.imread(os.path.join(C.SAMPLED_DIR, name))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dets = _ground_image(proc, model, rgb)

            yolo_lines = []
            for box, score, phrase in dets:
                cls = _phrase_to_class(phrase)
                if cls is None:
                    continue
                sam_iou = None
                if use_sam:
                    box, sam_iou = _refine_with_sam(sam, bgr, box)
                jf.write(json.dumps({
                    "image": name, "class": cls, "xyxy": [round(v, 1) for v in box],
                    "gd_conf": round(score, 4), "gd_phrase": phrase,
                    "sam_iou": (round(sam_iou, 4) if sam_iou is not None else None),
                    "img_w": w, "img_h": h,
                }) + "\n")
                cx, cy, bw, bh = _to_yolo(box, w, h)
                yolo_lines.append(f"{C.CLASS_TO_ID[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                n_det += 1

            with open(os.path.join(labels_dir, name.replace(C.IMG_EXT, ".txt")), "w") as lf:
                lf.write("\n".join(yolo_lines))
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(images)} images, {n_det} detections")

    print(f"\nDone. {n_det} detections across {len(images)} images.")
    print(f"Predictions: {C.PREDICTIONS_JSONL}")
    print(f"Raw YOLO labels: {labels_dir}")


if __name__ == "__main__":
    main()
