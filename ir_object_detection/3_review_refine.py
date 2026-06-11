"""
Stage 3 — AI-assisted review & refinement of the auto-generated labels.

Two automated passes that *propose* refinements; the human-validation gate in
Stage 4 / README is what ultimately accepts them.

A. Heuristic pass (free, deterministic)
   - class-wise NMS to merge duplicate detections
   - hand/face keypoint gating (reused logic from pipelines/vit_transformer/infer.py):
     drop detections whose center is not near a driver keypoint (skipped when no
     person is detected, rather than dropping everything)
   - confidence bands: >=CONF_HIGH auto-accept, <CONF_LOW drop, between -> VLM

B. VLM pass (Claude vision, Anthropic SDK)
   - crops each uncertain detection (+context) and asks the model to confirm /
     relabel the category among phone|bottle|cup|food|none, with a confidence
   - drops false positives, reclassifies, and flags low-confidence verdicts

Outputs:
  refined/labels/*.txt   clean YOLO labels (5-col)
  refined/review.jsonl   every kept detection + how it was decided / VLM verdict
  refined/flagged/       crops + sidecar for detections needing a human

Run: python ir_object_detection/3_review_refine.py [--no-vlm] [--limit N]
"""

import argparse
import base64
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402


# ─── geometry helpers ────────────────────────────────────────────────────────
def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms(dets, iou_thr):
    """Greedy NMS within a list of dets (each has xyxy + gd_conf). Keeps highest conf."""
    kept = []
    for d in sorted(dets, key=lambda x: x["gd_conf"], reverse=True):
        if all(_iou(d["xyxy"], k["xyxy"]) < iou_thr for k in kept):
            kept.append(d)
    return kept


# ─── keypoint gating (reused from infer.py) ──────────────────────────────────
def _keypoints(bgr, pose_model):
    res = pose_model(bgr, conf=C.POSE_CONF, verbose=False)
    if not res or res[0].keypoints is None or len(res[0].boxes) == 0:
        return np.empty((0, 2))
    best = int(np.argmax(res[0].boxes.conf.cpu().numpy()))
    kp = res[0].keypoints.data.cpu().numpy()[best]  # (17, 3)
    pts = [(x, y) for i, (x, y, c) in enumerate(kp)
           if i in C.RELEVANT_KP and c >= C.KP_CONF]
    return np.array(pts, dtype=np.float32) if pts else np.empty((0, 2))


def _near_keypoint(box, kps, radius):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if len(kps) == 0:
        return True  # no person detected -> don't gate (skip, don't drop)
    return bool(np.min(np.hypot(kps[:, 0] - cx, kps[:, 1] - cy)) <= radius)


# ─── VLM verdict ─────────────────────────────────────────────────────────────
def _crop_b64(bgr, box, pad):
    h, w = bgr.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1 = max(0, int(box[0] - pad * bw)); y1 = max(0, int(box[1] - pad * bh))
    x2 = min(w, int(box[2] + pad * bw)); y2 = min(h, int(box[3] + pad * bh))
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = bgr
    ok, buf = cv2.imencode(".png", crop)
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def _make_vlm_reviewer():
    """Return a fn(crop_b64, proposed_class) -> (category, confidence, reason) or None if unavailable."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — skipping VLM pass; uncertain dets flagged for humans.")
        return None
    try:
        import anthropic
    except ImportError:
        print("anthropic SDK not installed — skipping VLM pass; uncertain dets flagged for humans.")
        return None

    client = anthropic.Anthropic()
    schema = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": C.CLASSES + ["none"]},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["category", "confidence", "reason"],
        "additionalProperties": False,
    }
    prompt = (
        "This is a cropped region from an infrared (grayscale) in-car camera frame. "
        f"A detector proposed it contains a '{{proposed}}'. Classify the main object as one of: "
        f"{', '.join(C.CLASSES)}, or 'none' if there is no such object (false positive). "
        "Give a confidence 0-1 and a one-line reason."
    )

    def review(crop_b64, proposed_class):
        try:
            resp = client.messages.create(
                model=C.VLM_MODEL,
                max_tokens=512,
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64",
                                                     "media_type": "image/png", "data": crop_b64}},
                        {"type": "text", "text": prompt.format(proposed=proposed_class)},
                    ],
                }],
            )
            text = next(b.text for b in resp.content if b.type == "text")
            v = json.loads(text)
            return v["category"], float(v["confidence"]), v.get("reason", "")
        except Exception as e:
            print(f"  VLM error: {e}")
            return None

    return review


def _to_yolo(box, w, h):
    return ((box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h,
            (box[2] - box[0]) / w, (box[3] - box[1]) / h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vlm", action="store_true", help="skip the VLM pass (flag uncertain only)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # group predictions by image
    by_image = defaultdict(list)
    with open(C.PREDICTIONS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            by_image[d["image"]].append(d)
    images = sorted(by_image)
    if args.limit:
        images = images[: args.limit]

    from ultralytics import YOLO
    pose = YOLO(os.path.join(C.DATA_ROOT, C.POSE_WEIGHTS))
    review = None if args.no_vlm else _make_vlm_reviewer()

    os.makedirs(C.REFINED_LABELS_DIR, exist_ok=True)
    os.makedirs(C.FLAGGED_DIR, exist_ok=True)
    stats = defaultdict(int)

    with open(C.REVIEW_JSONL, "w") as rf:
        for name in images:
            bgr = cv2.imread(os.path.join(C.SAMPLED_DIR, name))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            kps = _keypoints(bgr, pose)
            radius = C.OBJ_KP_RADIUS_FRAC * max(h, w)

            # NMS per class
            per_class = defaultdict(list)
            for d in by_image[name]:
                per_class[d["class"]].append(d)
            deduped = []
            for cls, ds in per_class.items():
                deduped.extend(_nms(ds, C.NMS_IOU))
            stats["raw"] += len(by_image[name])
            stats["after_nms"] += len(deduped)

            kept_labels, flagged = [], []
            for d in deduped:
                box, conf, cls = d["xyxy"], d["gd_conf"], d["class"]

                if not _near_keypoint(box, kps, radius):
                    d["decision"] = "drop_keypoint_gate"; stats["drop_keypoint"] += 1
                    rf.write(json.dumps(d) + "\n"); continue
                if conf < C.CONF_LOW:
                    d["decision"] = "drop_low_conf"; stats["drop_low"] += 1
                    rf.write(json.dumps(d) + "\n"); continue

                if conf >= C.CONF_HIGH:
                    d["decision"] = "accept_high_conf"; stats["accept_high"] += 1
                else:
                    # uncertain band
                    if review is not None:
                        verdict = review(_crop_b64(bgr, box, C.VLM_CROP_PAD), cls)
                        if verdict is None:
                            d["decision"] = "flag_vlm_error"; flagged.append(d); stats["flag"] += 1
                            rf.write(json.dumps(d) + "\n"); continue
                        category, vconf, reason = verdict
                        d["vlm"] = {"category": category, "confidence": vconf, "reason": reason}
                        if category == "none":
                            d["decision"] = "drop_vlm_fp"; stats["drop_vlm"] += 1
                            rf.write(json.dumps(d) + "\n"); continue
                        if vconf < C.VLM_FLAG_CONF:
                            d["decision"] = "flag_vlm_uncertain"; flagged.append(d); stats["flag"] += 1
                            rf.write(json.dumps(d) + "\n"); continue
                        cls = category  # VLM may relabel
                        d["decision"] = "accept_vlm"; stats["accept_vlm"] += 1
                    else:
                        d["decision"] = "flag_no_vlm"; flagged.append(d); stats["flag"] += 1
                        rf.write(json.dumps(d) + "\n"); continue

                cx, cy, bw, bh = _to_yolo(box, w, h)
                kept_labels.append(f"{C.CLASS_TO_ID[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                rf.write(json.dumps(d) + "\n")

            with open(os.path.join(C.REFINED_LABELS_DIR, name.replace(C.IMG_EXT, ".txt")), "w") as lf:
                lf.write("\n".join(kept_labels))

            # write flagged crops for human review
            for i, d in enumerate(flagged):
                x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
                crop = bgr[max(0, y1):y2, max(0, x1):x2]
                if crop.size:
                    cv2.imwrite(os.path.join(C.FLAGGED_DIR, name.replace(C.IMG_EXT, f"__f{i}_{d['class']}.jpg")), crop)

    print("\nReview summary:")
    for k in ["raw", "after_nms", "drop_keypoint", "drop_low", "drop_vlm",
              "accept_high", "accept_vlm", "flag"]:
        print(f"  {k:14s}: {stats[k]}")
    print(f"\nRefined labels: {C.REFINED_LABELS_DIR}")
    print(f"Review log:     {C.REVIEW_JSONL}")
    print(f"Flagged crops:  {C.FLAGGED_DIR}  (review 100% of these — see README)")


if __name__ == "__main__":
    main()
