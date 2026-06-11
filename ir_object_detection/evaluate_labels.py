"""
Measure auto-label quality against a hand-labeled gold set.

Greedy IoU matching per class between predicted YOLO labels (from stage 3) and
gold YOLO labels you create by hand. Reports overall + per-class precision/recall
so you can apply the training gate (precision >= 0.90 and recall >= 0.80).

precision = TP / (TP + FP)   — of the boxes we drew, how many were real
recall    = TP / (TP + FN)   — of the real boxes, how many we found

Run:
  python ir_object_detection/evaluate_labels.py \
      --gold  ir_object_detection/gold/labels \
      --pred  ir_object_detection/refined/labels \
      --images ir_object_detection/sampled_frames \
      --iou 0.5
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402


def _load(path, w, h):
    """YOLO txt -> list of (cls, x1, y1, x2, y2) in pixels."""
    out = []
    if not os.path.exists(path):
        return out
    for ln in open(path).read().splitlines():
        if not ln.strip():
            continue
        c, cx, cy, bw, bh = ln.split()[:5]
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        out.append((int(c), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return out


def _iou(a, b):
    ix1, iy1 = max(a[1], b[1]), max(a[2], b[2])
    ix2, iy2 = min(a[3], b[3]), min(a[4], b[4])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[3] - a[1]) * (a[4] - a[2])
    ab = (b[3] - b[1]) * (b[4] - b[2])
    return inter / (aa + ab - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="dir of hand-labeled YOLO .txt (ground truth)")
    ap.add_argument("--pred", default=C.REFINED_LABELS_DIR, help="dir of predicted YOLO .txt")
    ap.add_argument("--images", default=C.SAMPLED_DIR, help="dir of the images (for pixel dims)")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    stems = sorted(f[:-4] for f in os.listdir(args.gold) if f.endswith(".txt"))
    if not stems:
        raise SystemExit(f"No gold .txt files in {args.gold}")

    tp = fp = fn = 0
    per = {c: [0, 0, 0] for c in range(len(C.CLASSES))}  # cls -> [tp, fp, fn]

    for stem in stems:
        img = cv2.imread(os.path.join(args.images, stem + C.IMG_EXT))
        h, w = (img.shape[:2] if img is not None else (1024, 1280))
        gold = _load(os.path.join(args.gold, stem + ".txt"), w, h)
        pred = _load(os.path.join(args.pred, stem + ".txt"), w, h)

        matched = set()
        for p in pred:
            best_j, best_iou = -1, args.iou
            for j, g in enumerate(gold):
                if j in matched or g[0] != p[0]:
                    continue
                v = _iou(p, g)
                if v >= best_iou:
                    best_j, best_iou = j, v
            if best_j >= 0:
                matched.add(best_j); tp += 1; per[p[0]][0] += 1
            else:
                fp += 1; per[p[0]][1] += 1
        for j, g in enumerate(gold):
            if j not in matched:
                fn += 1; per[g[0]][2] += 1

    def pr(t, f, n):
        p = t / (t + f) if (t + f) else 0.0
        r = t / (t + n) if (t + n) else 0.0
        return p, r

    p, r = pr(tp, fp, fn)
    print(f"\nGold images: {len(stems)}  |  IoU>={args.iou}")
    print(f"{'class':8s} {'TP':>4} {'FP':>4} {'FN':>4} {'prec':>6} {'rec':>6}")
    for c in range(len(C.CLASSES)):
        t, f, n = per[c]
        pc, rc = pr(t, f, n)
        print(f"{C.CLASSES[c]:8s} {t:4d} {f:4d} {n:4d} {pc:6.2f} {rc:6.2f}")
    print(f"{'OVERALL':8s} {tp:4d} {fp:4d} {fn:4d} {p:6.2f} {r:6.2f}")

    gate = p >= 0.90 and r >= 0.80
    print(f"\nGate (prec>=0.90 & rec>=0.80): {'PASS — ok to train' if gate else 'FAIL — fix before training'}")
    if not gate and r < 0.80:
        print("  recall low -> lower GD_BOX_THRESHOLD in config.py so GroundingDINO proposes more.")
    if not gate and p < 0.90:
        print("  precision low -> raise CONF_LOW / tighten VLM, or review accepts.")


if __name__ == "__main__":
    main()
