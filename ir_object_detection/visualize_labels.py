"""
Render bounding boxes onto the IR frames for visual inspection.

Writes annotated images + a contact sheet into ir_object_detection/viz/:
  viz/accepted/   kept labels (refined/labels) drawn on the frames
  viz/dropped/    detections Opus/heuristics rejected (from review.jsonl)  [--dropped]
  viz/_accepted_contact.jpg / _dropped_contact.jpg

Run:
  python ir_object_detection/visualize_labels.py                 # accepted labels
  python ir_object_detection/visualize_labels.py --dropped       # also render rejected dets
  python ir_object_detection/visualize_labels.py --labels ir_object_detection/gold/labels  # any YOLO dir
"""

import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

VIZ_DIR = os.path.join(C.PKG_DIR, "viz")
COLORS = {0: (0, 0, 255), 1: (0, 200, 0), 2: (255, 150, 0), 3: (0, 255, 255)}  # BGR


def _contact_sheet(files, out_path, cell=420, cols=3):
    if not files:
        return
    rows = math.ceil(len(files) / cols)
    sheet = np.zeros((rows * cell, cols * cell, 3), dtype="uint8")
    for i, f in enumerate(files):
        im = cv2.resize(cv2.imread(f), (cell, cell))
        r, c = divmod(i, cols)
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = im
    cv2.imwrite(out_path, sheet)


def render_accepted(labels_dir, images_dir):
    out = os.path.join(VIZ_DIR, "accepted")
    os.makedirs(out, exist_ok=True)
    n = 0
    for txt in sorted(glob.glob(labels_dir + "/*.txt")):
        rows = [r.split() for r in open(txt).read().splitlines() if r.strip()]
        if not rows:
            continue
        name = os.path.basename(txt).replace(".txt", C.IMG_EXT)
        img = cv2.imread(os.path.join(images_dir, name))
        if img is None:
            continue
        h, w = img.shape[:2]
        for r in rows:
            c = int(r[0]); cx, cy, bw, bh = [float(x) for x in r[1:5]]
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            col = COLORS.get(c, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
            cv2.putText(img, C.CLASSES[c], (x1, max(12, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        cv2.imwrite(os.path.join(out, name), img); n += 1
    _contact_sheet(sorted(glob.glob(out + "/*" + C.IMG_EXT)),
                   os.path.join(VIZ_DIR, "_accepted_contact.jpg"))
    print(f"accepted: {n} images -> {out}")


def render_dropped(images_dir):
    if not os.path.exists(C.REVIEW_JSONL):
        print("no review.jsonl — run stage 3 first; skipping dropped")
        return
    from collections import defaultdict
    drops = defaultdict(list)
    for d in map(json.loads, open(C.REVIEW_JSONL)):
        if str(d.get("decision", "")).startswith("drop"):
            drops[d["image"]].append(d)
    out = os.path.join(VIZ_DIR, "dropped")
    os.makedirs(out, exist_ok=True)
    n = 0
    for name, dets in sorted(drops.items()):
        img = cv2.imread(os.path.join(images_dir, name))
        if img is None:
            continue
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            tag = f"{d['class']} {d.get('gd_conf', 0):.2f} {d.get('decision', '')[5:]}"
            cv2.putText(img, tag, (x1, max(12, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(out, name), img); n += 1
    _contact_sheet(sorted(glob.glob(out + "/*" + C.IMG_EXT)),
                   os.path.join(VIZ_DIR, "_dropped_contact.jpg"))
    print(f"dropped:  {n} images -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=C.REFINED_LABELS_DIR, help="YOLO label dir to render")
    ap.add_argument("--images", default=C.SAMPLED_DIR, help="image dir")
    ap.add_argument("--dropped", action="store_true", help="also render rejected detections")
    args = ap.parse_args()

    os.makedirs(VIZ_DIR, exist_ok=True)
    render_accepted(args.labels, args.images)
    if args.dropped:
        render_dropped(args.images)
    print(f"\nViz folder: {VIZ_DIR}")
    print("Open the *_contact.jpg files for a quick overview.")


if __name__ == "__main__":
    main()
