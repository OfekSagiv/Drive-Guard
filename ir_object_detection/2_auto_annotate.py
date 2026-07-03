"""
Stage 2 — Auto-annotate sampled IR frames with GroundingDINO (+ optional SAM2).

This stage OVER-GENERATES candidates: Opus (Stage 3) can only review boxes that GD
proposes, so the goal here is candidate COVERAGE, not clean labels. Boxes are kept at
permissive per-class CANDIDATE thresholds; Stage 3 does the precision cleanup
(spatial gate -> NMS -> per-class accept band -> Opus -> manual flags).

* GroundingDINO (open-vocabulary) proposes boxes for config.GROUNDING_PROMPT.
* SAM2 (via ultralytics) optionally tightens each box from its segmentation mask.
* Predicted phrases are mapped onto the taxonomy (config.PHRASE_TO_CLASS).
* Each kept box is tagged with its band: "accept" (>= GD_ACCEPT_BOX_THRESHOLD) or
  "review" ([candidate, accept)) — Stage 3 uses this AFTER its spatial gate.

Outputs (one candidate per line):
  raw_predictions/predictions.jsonl  — image, class, xyxy(px), gd_conf, gd_phrase, band, sam_iou
  raw_predictions/labels/*.txt       — raw YOLO preview labels (ONLY for frames with candidates)
  raw_predictions/no_candidate_frames.txt — frames GD found nothing in (NOT written as empty
                                            labels, so missed objects don't become negatives)

Run: python ir_object_detection/2_auto_annotate.py [--limit N] [--no-sam] [--viz] [--enhance]
  --viz     draws the detected boxes (color per class) onto each image into viz/detections/.
  --enhance runs detection on a CLAHE+brightness copy (recovers faint objects in dark
            IR — esp. drink; trades some console false positives that Stage 3 gates).
            The stored dataset image stays the ORIGINAL; only detection sees the copy.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# BGR colors per class for --viz overlays.
VIZ_COLORS = {"phone": (0, 0, 255), "drink": (0, 255, 0), "food": (0, 255, 255)}


def _enhance(bgr):
    """CLAHE on luminance + mild gamma brighten — surfaces faint objects in dark IR.
    Used only for DETECTION; the stored dataset image stays the original."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    eq = np.clip(((eq / 255.0) ** 0.8) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


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
        threshold=C.GD_MIN_CANDIDATE, text_threshold=C.GD_TEXT_THRESHOLD,
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
    ap.add_argument("--viz", action="store_true", help="draw detected boxes onto images into viz/detections/")
    ap.add_argument("--enhance", action="store_true",
                    help="CLAHE+brightness on a detection-only copy (recovers faint IR objects)")
    args = ap.parse_args()

    images = sorted(f for f in os.listdir(C.SAMPLED_DIR) if f.endswith(C.IMG_EXT))
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images in {C.SAMPLED_DIR}; run stage 1 first.")

    print(f"Device: {DEVICE} | images: {len(images)}" + ("  | --enhance (CLAHE+brightness)" if args.enhance else ""))
    proc, model = _load_grounding()
    use_sam = C.USE_SAM2 and not args.no_sam
    sam = None
    if use_sam:
        from ultralytics import SAM
        sam = SAM(C.SAM2_WEIGHTS)

    os.makedirs(C.RAW_PRED_DIR, exist_ok=True)
    labels_dir = os.path.join(C.RAW_PRED_DIR, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    viz_dir = os.path.join(C.PKG_DIR, "viz", "detections")
    if args.viz:
        os.makedirs(viz_dir, exist_ok=True)

    n_det, n_accept, n_review, no_candidate = 0, 0, 0, []
    with open(C.PREDICTIONS_JSONL, "w") as jf:
        for i, name in enumerate(images):
            bgr = cv2.imread(os.path.join(C.SAMPLED_DIR, name))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            det_img = _enhance(bgr) if args.enhance else bgr  # detection-only; stored image stays original
            rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
            dets = _ground_image(proc, model, rgb)

            yolo_lines = []
            viz_img = det_img.copy() if args.viz else None  # overlay on what GD saw
            for box, score, phrase in dets:
                cls = _phrase_to_class(phrase)
                if cls is None:
                    continue
                # CANDIDATE gate (permissive) — over-generate; Stage 3 does precision.
                if score < C.GD_CANDIDATE_BOX_THRESHOLD.get(cls, C.GD_MIN_CANDIDATE):
                    continue
                # Tag the accept/review band (Stage 3 decides auto-accept vs Opus/manual,
                # AFTER its spatial gate). Metadata only — nothing is dropped here.
                band = "accept" if score >= C.GD_ACCEPT_BOX_THRESHOLD.get(cls, C.CONF_HIGH) else "review"
                n_accept += band == "accept"; n_review += band == "review"
                sam_iou = None
                if use_sam:
                    box, sam_iou = _refine_with_sam(sam, det_img, box)
                jf.write(json.dumps({
                    "image": name, "class": cls, "xyxy": [round(v, 1) for v in box],
                    "gd_conf": round(score, 4), "gd_phrase": phrase, "band": band,
                    "sam_iou": (round(sam_iou, 4) if sam_iou is not None else None),
                    "img_w": w, "img_h": h,
                }) + "\n")
                cx, cy, bw, bh = _to_yolo(box, w, h)
                yolo_lines.append(f"{C.CLASS_TO_ID[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                n_det += 1
                if viz_img is not None:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    color = VIZ_COLORS.get(cls, (255, 255, 255))
                    cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(viz_img, f"{cls} {score:.2f} {band[0]}", (x1, max(12, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # Only write a label file when there ARE candidates. A frame with no candidate
            # is logged, NOT written as an empty label — an empty YOLO file means "no
            # object here" and would teach the detector to suppress real (missed) objects.
            if yolo_lines:
                with open(os.path.join(labels_dir, name.replace(C.IMG_EXT, ".txt")), "w") as lf:
                    lf.write("\n".join(yolo_lines))
            else:
                no_candidate.append(name)
            if viz_img is not None:
                cv2.imwrite(os.path.join(viz_dir, name), viz_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(images)} images, {n_det} candidates")

    nc_path = os.path.join(C.RAW_PRED_DIR, "no_candidate_frames.txt")
    with open(nc_path, "w") as f:
        f.write("\n".join(no_candidate))

    print(f"\nDone. {n_det} candidate boxes across {len(images)} images "
          f"({n_accept} accept-band, {n_review} review-band).")
    print(f"No-candidate frames (NOT labeled empty): {len(no_candidate)} -> {nc_path}")
    print(f"Predictions (all candidates + metadata): {C.PREDICTIONS_JSONL}")
    print(f"Raw YOLO labels: {labels_dir}")
    if args.viz:
        print(f"Detection overlays: {viz_dir}")


if __name__ == "__main__":
    main()
