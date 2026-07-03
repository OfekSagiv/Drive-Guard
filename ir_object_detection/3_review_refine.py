"""
Stage 3 — review & refine the Stage 2 candidate boxes into clean labels.

Stage 2 OVER-GENERATES candidates (permissive per-class thresholds, each tagged with
a `band`: "accept" or "review"). Stage 3 does the precision cleanup, in this order:

  1. NMS              — class-wise dedup of candidates.
  2. Keypoint gate    — drop boxes not near the driver's hands/face (REVIEW_KP_RADIUS_FRAC,
                        re-tuned for ROI crops). Skipped (kept) if no person is detected.
  3. Band decision    — a GATED box that is:
                          accept-band -> AUTO-ACCEPT (clean, high-confidence).
                          review-band -> send to OPUS (confirm / relabel / drop / flag).
  4. Manual flags     — review-band with no VLM, Opus-uncertain verdicts, AND object-
                        activity frames that yielded NO accepted/reviewed box (missed
                        objects, incl. the Stage-2 no-candidate frames).
  5. Negatives        — a `safe` frame is kept as a negative ONLY if no accepted/reviewed
                        object box remains on it.

The accept/review band is decided in Stage 2 by GD_ACCEPT_BOX_THRESHOLD; the gate runs
FIRST, so high-confidence off-driver console FPs are dropped before the accept bar applies.

Outputs (under IR_OUT_DIR):
  refined/labels/*.txt   clean YOLO labels (accepted + Opus-accepted boxes)
  refined/review.jsonl   every candidate + how it was decided
  refined/flagged.txt    frames needing a human (missed objects + uncertain)
  refined/negatives.txt  safe frames kept as clean negatives

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
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402


# ─── --viz colors (by Stage 3 decision) ──────────────────────────────────────
def _dec_color(dec):
    if dec in ("accept_auto", "accept_vlm"):
        return (0, 255, 0)          # green  — auto/Opus accepted
    if dec == "drop_gate":
        return (0, 0, 255)          # red    — removed by keypoint gate
    return (0, 255, 255)            # yellow — review-band -> Opus (or its verdict)


def _build_contact(viz_dir, obj_class, out, cell=190, cols=10):
    files = [f for f in os.listdir(viz_dir) if f.endswith(C.IMG_EXT)]
    files.sort(key=lambda n: (["phone", "drink", "food", "safe", "?"].index(obj_class.get(n, "?")), n))
    tiles = []
    for n in files:
        img = cv2.imread(os.path.join(viz_dir, n))
        if img is None:
            continue
        h, w = img.shape[:2]; s = min(h, w)
        t = cv2.resize(img[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s], (cell, cell))
        cv2.putText(t, obj_class.get(n, "?")[:4], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        tiles.append(t)
    if not tiles:
        return
    rows = (len(tiles) + cols - 1) // cols
    g = np.zeros((rows*cell, cols*cell, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols); g[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = t
    cv2.imwrite(out, g, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


# ─── geometry ────────────────────────────────────────────────────────────────
def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _nms(dets, iou_thr):
    kept = []
    for d in sorted(dets, key=lambda x: x["gd_conf"], reverse=True):
        if all(_iou(d["xyxy"], k["xyxy"]) < iou_thr for k in kept):
            kept.append(d)
    return kept


# ─── keypoint gate (reused from pipelines/vit_transformer/infer.py) ──────────
def _keypoints(bgr, pose):
    r = pose(bgr, conf=C.POSE_CONF, verbose=False)
    if not r or r[0].keypoints is None or len(r[0].boxes) == 0:
        return np.empty((0, 2))
    best = int(np.argmax(r[0].boxes.conf.cpu().numpy()))
    kp = r[0].keypoints.data.cpu().numpy()[best]
    pts = [(x, y) for i, (x, y, c) in enumerate(kp) if i in C.RELEVANT_KP and c >= C.KP_CONF]
    return np.array(pts, np.float32) if pts else np.empty((0, 2))


def _near(box, kps, radius):
    """Point-to-box gate: keep if a relevant keypoint is INSIDE the box, or within
    `radius` of the box (point-to-rectangle distance). Robust to large boxes
    (wrappers/food spanning hand-to-face) where the box CENTER would read as far.
    Skip-keep when no person is detected."""
    if len(kps) == 0:
        return True
    # point-to-rectangle distance per keypoint (0 if inside the box)
    dx = np.maximum.reduce([box[0] - kps[:, 0], kps[:, 0] - box[2], np.zeros(len(kps))])
    dy = np.maximum.reduce([box[1] - kps[:, 1], kps[:, 1] - box[3], np.zeros(len(kps))])
    return bool(np.min(np.hypot(dx, dy)) <= radius)


# ─── Opus review for review-band boxes ───────────────────────────────────────
def _crop_b64(bgr, box, pad):
    h, w = bgr.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1 = max(0, int(box[0] - pad*bw)); y1 = max(0, int(box[1] - pad*bh))
    x2 = min(w, int(box[2] + pad*bw)); y2 = min(h, int(box[3] + pad*bh))
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = bgr
    ok, buf = cv2.imencode(".png", crop)
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def _make_reviewer():
    """fn(crop_b64, proposed) -> (category, conf, reason) or None if VLM unavailable."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — review-band boxes will be FLAGGED, not Opus-reviewed.")
        return None
    try:
        import anthropic
    except ImportError:
        print("anthropic SDK missing — review-band boxes will be FLAGGED.")
        return None
    client = anthropic.Anthropic()
    schema = {"type": "object", "properties": {
        "category": {"type": "string", "enum": C.CLASSES + ["none"]},
        "confidence": {"type": "number"}, "reason": {"type": "string"}},
        "required": ["category", "confidence", "reason"], "additionalProperties": False}
    prompt = ("This is a cropped region from an infrared (grayscale) in-car camera frame. "
              "A detector proposed it contains a '{proposed}'. Classify the main object as one "
              f"of: {', '.join(C.CLASSES)}, or 'none' if there is no such object (false positive). "
              "Give a confidence 0-1 and a one-line reason.")

    def review(crop_b64, proposed):
        try:
            resp = client.messages.create(
                model=C.VLM_MODEL, max_tokens=512,
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": crop_b64}},
                    {"type": "text", "text": prompt.format(proposed=proposed)}]}])
            v = json.loads(next(b.text for b in resp.content if b.type == "text"))
            return v["category"], float(v["confidence"]), v.get("reason", "")
        except Exception as e:
            print(f"  VLM error: {e}")
            return None
    return review


def _to_yolo(box, w, h):
    return ((box[0]+box[2])/2/w, (box[1]+box[3])/2/h, (box[2]-box[0])/w, (box[3]-box[1])/h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vlm", action="store_true", help="flag review-band boxes instead of Opus review")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--viz", action="store_true",
                    help="render per-decision overlays (red=gated, yellow=Opus, green=accepted) + contact sheet")
    args = ap.parse_args()

    by_image = defaultdict(list)
    with open(C.PREDICTIONS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            by_image[d["image"]].append(d)

    # obj_class per frame (activity proxy) for missed-object flags + safe negatives
    obj_class = {}
    if os.path.exists(C.FROZEN_MANIFEST):
        m = pd.read_csv(C.FROZEN_MANIFEST)
        obj_class = dict(zip(m.image, m.obj_class))
    nc_path = os.path.join(C.RAW_PRED_DIR, "no_candidate_frames.txt")
    no_candidate = [l.strip() for l in open(nc_path)] if os.path.exists(nc_path) else []

    # every frame we must account for = candidate frames + no-candidate frames
    all_frames = sorted(set(by_image) | set(no_candidate))
    if args.limit:
        all_frames = all_frames[:args.limit]

    from ultralytics import YOLO
    pose = YOLO(os.path.join(C.DATA_ROOT, C.POSE_WEIGHTS))
    review = None if args.no_vlm else _make_reviewer()

    os.makedirs(C.REFINED_LABELS_DIR, exist_ok=True)
    os.makedirs(C.REFINED_DIR, exist_ok=True)
    viz_dir = os.path.join(C.PKG_DIR, "viz", "stage3")
    if args.viz:
        os.makedirs(viz_dir, exist_ok=True)
    stats = defaultdict(int)
    flagged, negatives, fp_auto = [], [], []

    with open(C.REVIEW_JSONL, "w") as rf:
        for name in all_frames:
            cls_gt = obj_class.get(name, "?")
            cands = by_image.get(name, [])

            # no-candidate frame: missed object (flag) or clean safe negative
            if not cands:
                if cls_gt in C.CLASSES:
                    flagged.append(f"{name}\tno_candidate_missed_{cls_gt}"); stats["flag_missed"] += 1
                    tag, col = f"MISSED {cls_gt}", (0, 0, 255)
                elif cls_gt == "safe":
                    negatives.append(name); stats["neg_safe"] += 1
                    tag, col = "neg (safe)", (160, 160, 160)
                else:
                    tag, col = "no candidate", (160, 160, 160)
                if args.viz:
                    im = cv2.imread(os.path.join(C.SAMPLED_DIR, name))
                    if im is not None:
                        cv2.putText(im, tag, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                        cv2.imwrite(os.path.join(viz_dir, name), im, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                continue

            bgr = cv2.imread(os.path.join(C.SAMPLED_DIR, name))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            kps = _keypoints(bgr, pose)
            radius = C.REVIEW_KP_RADIUS_FRAC * max(h, w)

            per_class = defaultdict(list)
            for d in cands:
                per_class[d["class"]].append(d)
            deduped = []
            for c, ds in per_class.items():
                deduped.extend(_nms(ds, C.NMS_IOU))
            stats["candidates"] += len(cands)
            stats["after_nms"] += len(deduped)

            kept_labels = []
            for d in deduped:
                box, cls, band = d["xyxy"], d["class"], d.get("band", "review")

                if not _near(box, kps, radius):
                    d["decision"] = "drop_gate"; stats["drop_gate"] += 1
                    rf.write(json.dumps(d) + "\n"); continue

                if band == "accept":
                    d["decision"] = "accept_auto"; stats["accept_auto"] += 1
                else:  # review-band -> Opus / flag
                    if review is None:
                        d["decision"] = "flag_review"; stats["to_opus"] += 1
                        flagged.append(f"{name}\treview_band_{cls}")
                        rf.write(json.dumps(d) + "\n"); continue
                    stats["to_opus"] += 1
                    verdict = review(_crop_b64(bgr, box, C.VLM_CROP_PAD), cls)
                    if verdict is None:
                        d["decision"] = "flag_vlm_error"; flagged.append(f"{name}\tvlm_error")
                        rf.write(json.dumps(d) + "\n"); continue
                    cat, vconf, reason = verdict
                    d["vlm"] = {"category": cat, "confidence": vconf, "reason": reason}
                    if cat == "none":
                        d["decision"] = "drop_vlm"; stats["drop_vlm"] += 1
                        rf.write(json.dumps(d) + "\n"); continue
                    if vconf < C.VLM_FLAG_CONF:
                        d["decision"] = "flag_vlm_uncertain"; flagged.append(f"{name}\tvlm_uncertain_{cat}")
                        rf.write(json.dumps(d) + "\n"); continue
                    cls = cat; d["decision"] = "accept_vlm"; stats["accept_vlm"] += 1

                cx, cy, bw, bh = _to_yolo(box, w, h)
                kept_labels.append(f"{C.CLASS_TO_ID[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                rf.write(json.dumps(d) + "\n")

            if args.viz:
                vis = bgr.copy()
                for d in deduped:
                    dec = d.get("decision", "")
                    bx = [int(v) for v in d["xyxy"]]
                    color = _dec_color(dec)
                    # suspected FP auto-accept: accepted but class != activity (or on a safe frame)
                    is_fp_auto = dec == "accept_auto" and (cls_gt == "safe" or
                                  (cls_gt in C.CLASSES and d["class"] != cls_gt))
                    thick = 3 if is_fp_auto else 2
                    cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), color, thick)
                    lab = f"{d['class']} {d['gd_conf']:.2f} {d.get('band','')[:3]}"
                    if is_fp_auto:
                        lab += " FP?"
                        cv2.rectangle(vis, (bx[0]-2, bx[1]-2), (bx[2]+2, bx[3]+2), (255, 0, 255), 1)
                        fp_auto.append(f"{name}\t{d['class']}\t{d['gd_conf']:.2f}\tgt={cls_gt}")
                    cv2.putText(vis, lab, (bx[0], max(12, bx[1]-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                cv2.putText(vis, f"gt:{cls_gt}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imwrite(os.path.join(viz_dir, name), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

            if kept_labels:
                with open(os.path.join(C.REFINED_LABELS_DIR, name.replace(C.IMG_EXT, ".txt")), "w") as lf:
                    lf.write("\n".join(kept_labels))
            else:
                # nothing survived: safe -> negative; object-activity -> flag missed
                if cls_gt == "safe":
                    negatives.append(name); stats["neg_safe"] += 1
                elif cls_gt in C.CLASSES:
                    flagged.append(f"{name}\tall_dropped_missed_{cls_gt}"); stats["flag_missed"] += 1

    with open(os.path.join(C.REFINED_DIR, "flagged.txt"), "w") as f:
        f.write("\n".join(flagged))
    with open(os.path.join(C.REFINED_DIR, "negatives.txt"), "w") as f:
        f.write("\n".join(negatives))

    print("\nStage 3 summary:")
    for k in ["candidates", "after_nms", "drop_gate", "accept_auto", "to_opus",
              "accept_vlm", "drop_vlm", "flag_missed", "neg_safe"]:
        print(f"  {k:14s}: {stats[k]}")
    print(f"  flagged (manual): {len(flagged)}  |  negatives (safe): {len(negatives)}")
    print(f"\nLabels:    {C.REFINED_LABELS_DIR}")
    print(f"Review:    {C.REVIEW_JSONL}")
    print(f"Flagged:   {os.path.join(C.REFINED_DIR, 'flagged.txt')}")
    print(f"Negatives: {os.path.join(C.REFINED_DIR, 'negatives.txt')}")

    if args.viz:
        sheet = os.path.join(C.PKG_DIR, "viz", "stage3_contact.jpg")
        _build_contact(viz_dir, obj_class, sheet)
        with open(os.path.join(C.REFINED_DIR, "fp_autoaccept.txt"), "w") as f:
            f.write("\n".join(fp_auto))
        print("\n--viz legend: red=gate-removed | yellow=Opus-review | green=accepted | magenta=suspected FP auto-accept")
        print(f"Overlays:  {viz_dir}")
        print(f"Contact:   {sheet}")
        print(f"Suspected FP auto-accepts: {len(fp_auto)} -> {os.path.join(C.REFINED_DIR, 'fp_autoaccept.txt')}")


if __name__ == "__main__":
    main()
