#!/usr/bin/env python3
"""
evaluate_infer.py — quantitative evaluation of infer.py's inference math.

Answers a deliberately neutral question: **does the object-detection fusion help
at all, and at what fusion weight W?**  (Not "confirm it helps.")

Method (tune on val, report on test — never tune W on the test set):
  1. Sweep W on the val split; pick W* = best macro-F1.
  2. Lock W*; report baseline (W=0) vs fused (W*) once on the test split:
     per-class precision/recall/F1, side-by-side confusion matrices, and an
     EXACT McNemar test on the prediction flips.

Faithfulness: this imports and runs infer.py's own functions
(crop_and_preprocess, load_spatial/temporal_model, get_person_keypoints,
detect_object_token, compute_object_prior, fuse_probs) over the labeled
16-frame ROI-crop clips in ds_driveguard_temporal_roi/{val,test}. Each clip is
exactly one WINDOW_FRAMES window, so it reproduces one infer.py decision.

CAVEATS — this is a controlled proxy, NOT a streaming replay:
  • Object signal differs from production: live infer.py runs the COCO detector on
    the FULL frame then gates to the driver ROI; here it runs on the pre-cropped
    384px ROI JPEG (different scale/context). Keypoint gating on a tight crop may
    fire even less than on full frames.
  • Bypasses live YOLO-pose ROI-locking and STEP frame sampling (no raw video with
    per-frame labels is available).
The numbers measure the fusion DECISION math on labeled windows; they do not
replay the full streaming pipeline.
"""

import argparse
import glob
import math
import os
import sys
import warnings
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch

# infer.py lives next to this file and is import-safe (all Drive downloads are
# inside functions / the __main__ block).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer  # noqa: E402
from infer import (  # noqa: E402
    CLASSES,
    WINDOW_FRAMES,
    OBJ_WEIGHTS,
    OBJ_KP_RADIUS_FRAC,
    crop_and_preprocess,
    load_spatial_model,
    load_temporal_model,
    get_person_keypoints,
    detect_object_token,
    compute_object_prior,
    fuse_probs,
    _ensure_downloaded,
)
from ultralytics import YOLO  # noqa: E402

NOTEBOOK_BASELINE_ACC = 0.94   # known end-to-end accuracy → baseline sanity check


# ──────────────────────────────────────────────────────────────────────────────
# Device (mirrors infer.run_inference)
# ──────────────────────────────────────────────────────────────────────────────
def pick_device():
    if torch.backends.mps.is_available():
        print("Device : MPS  (Apple Silicon)  — FP16")
        return torch.device('mps'), True
    if torch.cuda.is_available():
        print(f"Device : CUDA  ({torch.cuda.get_device_name(0)})  — FP16")
        return torch.device('cuda'), True
    print("Device : CPU  — FP32")
    return torch.device('cpu'), False


# ──────────────────────────────────────────────────────────────────────────────
# Dataset iteration
# ──────────────────────────────────────────────────────────────────────────────
def iter_clips(split_root: str, camera: str, limit: int | None):
    """Yield (label_idx, [16 frame paths]) for clips of `camera` in split_root."""
    per_class = Counter()
    for label_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(split_root, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [WARN] missing class dir: {cls_dir}")
            continue
        for seq_id in sorted(os.listdir(cls_dir)):
            if not seq_id.startswith(camera):
                continue
            if limit is not None and per_class[cls] >= limit:
                break
            frames = sorted(glob.glob(os.path.join(cls_dir, seq_id, '*.jpg')))[:WINDOW_FRAMES]
            if len(frames) != WINDOW_FRAMES:
                continue
            per_class[cls] += 1
            yield label_idx, frames
    print(f"  clips: " + ", ".join(f"{c}={per_class[c]}" for c in CLASSES))


# ──────────────────────────────────────────────────────────────────────────────
# Score one split: run the real inference path once per clip, reuse across all W
# ──────────────────────────────────────────────────────────────────────────────
def score_split(split_root, camera, sweep, models, device, use_fp16,
                use_kp_gate, limit):
    spatial_model, temporal_model, yolo_pose, yolo_obj = models
    y_true = []
    preds = {w: [] for w in sweep}          # W -> [pred_idx, ...]
    n_with_obj = 0                          # clips with ≥1 non-None token
    obj_confs = []                          # confidences of fired detections

    import time as _time
    t_start = _time.perf_counter()
    n_done = 0
    for label_idx, frame_paths in iter_clips(split_root, camera, limit):
        bgr_frames = [cv2.imread(p) for p in frame_paths]
        if any(f is None for f in bgr_frames):
            continue
        h, w = bgr_frames[0].shape[:2]
        roi = (0, 0, w, h)

        # ── Spatial → temporal (baseline softmax) ──────────────────────────────
        tensors = [crop_and_preprocess(f, roi) for f in bgr_frames]
        batch = torch.stack(tensors).to(device)            # (16, 3, 384, 384)
        if use_fp16:
            batch = batch.half()
        with torch.no_grad():
            feats = spatial_model(batch)                   # (16, 1152)
            logits = temporal_model(feats.unsqueeze(0))    # (1, 3)
        baseline = torch.softmax(logits, dim=-1).float().cpu().squeeze(0)

        # ── Object signal over the 16 frames (deployed gating) ─────────────────
        kp_radius = OBJ_KP_RADIUS_FRAC * max(w, h)
        obj_deque = deque(maxlen=WINDOW_FRAMES)
        fired = False
        for f in bgr_frames:
            kps = get_person_keypoints(f, yolo_pose) if use_kp_gate else None
            tok, _box, conf = detect_object_token(f, yolo_obj, roi, kps, kp_radius)
            obj_deque.append((tok, conf))
            if tok is not None:
                fired = True
                obj_confs.append(conf)
        if fired:
            n_with_obj += 1
        prior = compute_object_prior(obj_deque)

        # ── Fuse once per W (cheap; detector already ran) ──────────────────────
        y_true.append(label_idx)
        for wgt in sweep:
            fused = fuse_probs(baseline, prior, wgt)
            preds[wgt].append(int(fused.argmax()))

        n_done += 1
        if n_done % 25 == 0:
            rate = (_time.perf_counter() - t_start) / n_done
            print(f"    … {n_done} clips  ({rate:.2f}s/clip)", flush=True)

    n = len(y_true)
    print(f"  scored {n} clips | clips with ≥1 detection: {n_with_obj} "
          f"({100 * n_with_obj / max(n, 1):.1f}%) | "
          f"mean fired conf: {np.mean(obj_confs):.3f}" if obj_confs
          else f"  scored {n} clips | no detections fired")
    return np.array(y_true), {w: np.array(p) for w, p in preds.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def confusion_matrix(y_true, y_pred):
    k = len(CLASSES)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(y_true, y_pred):
    """Return (precision, recall, f1) arrays over classes from the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
    rec  = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)
    return prec, rec, f1


def metrics_row(y_true, y_pred):
    _, _, f1 = per_class_prf(y_true, y_pred)
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if len(y_true) else 0.0
    return {
        'acc':     acc,
        'macroF1': float(np.mean(f1)),
        'PhoneF1': float(f1[CLASSES.index('Phone')]),
        'DrinkF1': float(f1[CLASSES.index('Drink')]),
    }


def classification_report(y_true, y_pred):
    prec, rec, f1 = per_class_prf(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    support = cm.sum(axis=1)
    lines = [f"    {'class':>6} {'prec':>8} {'recall':>8} {'f1':>8} {'support':>8}"]
    for i, cls in enumerate(CLASSES):
        lines.append(f"    {cls:>6} {prec[i]:>8.4f} {rec[i]:>8.4f} {f1[i]:>8.4f} {support[i]:>8d}")
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if len(y_true) else 0.0
    lines.append(f"    {'macro':>6} {prec.mean():>8.4f} {rec.mean():>8.4f} "
                 f"{f1.mean():>8.4f} {support.sum():>8d}")
    lines.append(f"    {'acc':>6} {'':>8} {'':>8} {acc:>8.4f} {support.sum():>8d}")
    return "\n".join(lines)


def mcnemar_exact(y_true, pred_a, pred_b):
    """Exact (binomial) McNemar on correctness flips between two predictors."""
    a_ok = (pred_a == y_true)
    b_ok = (pred_b == y_true)
    b = int(np.sum(a_ok & ~b_ok))   # a correct, b wrong
    c = int(np.sum(~a_ok & b_ok))   # a wrong, b correct
    n = b + c
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        both_ok = int(np.sum(a_ok & b_ok))
        both_no = int(np.sum(~a_ok & ~b_ok))
        res = mcnemar([[both_ok, b], [c, both_no]], exact=True)
        p = float(res.pvalue)
    except Exception:
        # two-sided exact binomial with p=0.5
        if n == 0:
            p = 1.0
        else:
            k = min(b, c)
            tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
            p = min(1.0, 2.0 * tail)
    return b, c, p


def print_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  {title} (rows=true, cols=pred): {CLASSES}")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls:>6} " + " ".join(f"{v:5d}" for v in cm[i]))
    return cm


def save_confusion_png(cm_base, cm_fused, wstar, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [WARN] matplotlib unavailable, skipping PNG: {e}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cm, ttl in zip(axes, [cm_base, cm_fused],
                           ['Baseline (W=0)', f'Fused (W*={wstar})']):
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(ttl)
        ax.set_xticks(range(len(CLASSES)), CLASSES)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel('pred'); ax.set_ylabel('true')
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        color='black', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = os.path.join(out_dir, 'confusion_infer_eval.png')
    fig.savefig(path, dpi=120)
    print(f"\n  Confusion matrices → {path}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    default_ds = here.parent.parent / 'ds_driveguard_temporal_roi'
    ap.add_argument('--val_root',  default=str(default_ds / 'val'))
    ap.add_argument('--test_root', default=str(default_ds / 'test'))
    ap.add_argument('--camera', default=infer.CAMERA)
    ap.add_argument('--sweep', default='0.0,0.2,0.4,0.6,0.8,1.0,1.5')
    ap.add_argument('--obj_weights', default=OBJ_WEIGHTS)
    ap.add_argument('--no_kp_gate', action='store_true',
                    help='Score the ungated detector (diagnose keypoint-gate suppression)')
    ap.add_argument('--spatial_weights', default=None)
    ap.add_argument('--temporal_weights', default=None)
    ap.add_argument('--limit', type=int, default=None,
                    help='Cap clips per class per split (fast smoke run)')
    ap.add_argument('--out_dir', default=str(here))
    args = ap.parse_args()

    sweep = [round(float(x), 4) for x in args.sweep.split(',')]
    if 0.0 not in sweep:
        sweep = [0.0] + sweep
    sweep = sorted(set(sweep))

    print(__doc__.split('\n\n')[0])   # one-line title
    print("\n" + "─" * 70)
    print("CAVEAT: object signal here runs on the pre-cropped 384px ROI, not the")
    print("full frame as in production; ROI-lock and STEP sampling are bypassed.")
    print("This measures the fusion DECISION math, not a streaming replay.")
    print("─" * 70 + "\n")

    spatial_w  = args.spatial_weights  or _ensure_downloaded('vit_spatial_model_v1.pth')
    temporal_w = args.temporal_weights or _ensure_downloaded('temporal_head_model.pth')

    device, use_fp16 = pick_device()
    print("\nLoading models …")
    yolo_pose      = YOLO('yolov8n-pose.pt')
    yolo_obj       = YOLO(args.obj_weights)
    spatial_model  = load_spatial_model(spatial_w, device, use_fp16)
    temporal_model = load_temporal_model(temporal_w, device, use_fp16)
    models = (spatial_model, temporal_model, yolo_pose, yolo_obj)
    use_kp_gate = not args.no_kp_gate
    print(f"  keypoint gate: {'ON (deployed path)' if use_kp_gate else 'OFF (ungated)'}\n")

    # ── 1. Tune W on val ───────────────────────────────────────────────────────
    print("═" * 70)
    print(f"VAL sweep  (camera={args.camera})")
    print("═" * 70)
    yv, pv = score_split(args.val_root, args.camera, sweep, models,
                         device, use_fp16, use_kp_gate, args.limit)
    print(f"\n  {'W':>5} {'acc':>7} {'macroF1':>8} {'PhoneF1':>8} {'DrinkF1':>8}")
    best_w, best_macro = 0.0, -1.0
    for w in sweep:
        m = metrics_row(yv, pv[w])
        flag = ''
        if m['macroF1'] > best_macro:
            best_macro, best_w = m['macroF1'], w
        print(f"  {w:>5} {m['acc']:>7.4f} {m['macroF1']:>8.4f} "
              f"{m['PhoneF1']:>8.4f} {m['DrinkF1']:>8.4f}")
    print(f"\n  → W* = {best_w}  (best val macro-F1 = {best_macro:.4f})")

    # ── 2. Report on test with locked W* ───────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"TEST  baseline (W=0) vs fused (W*={best_w})   (camera={args.camera})")
    print("═" * 70)
    yt, pt = score_split(args.test_root, args.camera, sorted({0.0, best_w}), models,
                         device, use_fp16, use_kp_gate, args.limit)

    base_pred, fused_pred = pt[0.0], pt[best_w]

    base_acc = metrics_row(yt, base_pred)['acc']
    print(f"\n  Baseline test accuracy = {base_acc:.4f} "
          f"(notebook reference ≈ {NOTEBOOK_BASELINE_ACC})")
    if abs(base_acc - NOTEBOOK_BASELINE_ACC) > 0.05:
        print(f"  [WARN] baseline drifted >0.05 from reference — check weights/camera.")

    print("\n  ── Baseline (W=0) ──")
    print(classification_report(yt, base_pred))
    print(f"\n  ── Fused (W*={best_w}) ──")
    print(classification_report(yt, fused_pred))

    cm_b = print_confusion(yt, base_pred,  "Baseline confusion")
    cm_f = print_confusion(yt, fused_pred, "Fused confusion")
    save_confusion_png(cm_b, cm_f, best_w, args.out_dir)

    b, c, p = mcnemar_exact(yt, base_pred, fused_pred)
    print(f"\n  McNemar (exact): baseline→wrong/fused→right c={c}, "
          f"baseline→right/fused→wrong b={b}, discordant={b + c}, p={p:.4f}")
    delta = metrics_row(yt, fused_pred)['macroF1'] - metrics_row(yt, base_pred)['macroF1']
    verdict = ("HELPS" if delta > 0 and c > b else
               "HURTS" if delta < 0 else "NEUTRAL")
    print(f"  macro-F1 delta (fused − baseline) = {delta:+.4f}  →  fusion {verdict} "
          f"at W*={best_w}"
          + ("" if p < 0.05 else "  (NOT statistically significant)"))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
