"""
ROI variant extractor — crops a padded driver ROI instead of full frames.

Reuses the spatial pipeline's ROI logic verbatim (`get_square_box`, `crop_roi`,
and the shared `yolov8n-pose` detector from `extract_spatial_roi_ds.py`) rather
than implementing a new cropping method. The only knob added here is a larger,
configurable padding so nearby objects (hands, lap, center console, bottles /
cups / phones held away from the body) are preserved at the ROI boundary.

Modes:
  --preview N   extract N frames (stratified across obj_class) as side-by-side
                full-frame|ROI comparisons into viz/roi_preview/, and print ROI
                dimension statistics. No full extraction.
  (default)     extract every frame in the manifest as a padded ROI crop into
                <IR_SAMPLED_DIR or sampled_frames_roi>/.

Defaults (config.py): manifest = FROZEN_MANIFEST (manifest_codriver_roi.csv),
output = SAMPLED_DIR (sampled_frames_roi/, the dir stage 2 reads).

Run:
  python extract_roi_variant.py --preview 28     # eyeball ROI boxes + stats, no full run
  python extract_roi_variant.py                  # full extraction -> sampled_frames_roi/
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

# Reuse the EXISTING ROI logic from the spatial pipeline. The module loads the
# yolov8n-pose detector at import time relative to the repo root, so make the repo
# root the cwd and import path before importing it.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_PKG_DIR, ".."))
sys.path.insert(0, _PKG_DIR)
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import config as C  # noqa: E402
import extract_spatial_roi_ds as sp  # noqa: E402  (loads sp.DETECTOR, sp.DEVICE)


KP_MIN_SIDE_FRAC = 0.20   # reject keypoint boxes whose side < this * frame max-dim
                          # (a couple of clustered face keypoints can collapse the box)


def _keypoint_box(res, person, pad, h, w):
    """Square ROI around the object-relevant keypoints (face + elbows/wrists), so the
    crop centers on where phone/cup/food actually are — not the whole reclined body.
    Reuses C.RELEVANT_KP / C.KP_CONF and the shared get_square_box. Returns None when
    too few keypoints are confident OR the box collapses below KP_MIN_SIDE_FRAC
    (caller then falls back to the reliable person box)."""
    if res.keypoints is None:
        return None
    kp = res.keypoints.data.cpu().numpy()[person]  # (17, 3): x, y, conf
    pts = [(x, y) for i, (x, y, c) in enumerate(kp)
           if i in C.RELEVANT_KP and c >= C.KP_CONF]
    if len(pts) < 2:
        return None
    pts = np.array(pts, float)
    box = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    nx1, ny1, nx2, ny2 = sp.get_square_box(box, h, w, padding=pad)
    if (nx2 - nx1) < KP_MIN_SIDE_FRAC * max(h, w):
        return None  # degenerate (e.g. only clustered face points) -> use person box
    return nx1, ny1, nx2, ny2


def roi_box_for_frame(cap, frame_idx, pad, mode="keypoints", search=10):
    """Find a padded square driver ROI near `frame_idx`.

    Scans a few frames for the best pose detection (like
    extract_spatial_roi_ds.get_roi_box) and threads a configurable padding through
    the shared get_square_box. `mode`:
      - "keypoints": box around face + hands keypoints (object-focused, tighter)
      - "person":    box around the full person (the spatial pipeline's behavior)
    Keypoint mode falls back to the person box if too few keypoints are confident.
    Returns (x1,y1,x2,y2) or None (caller falls back to full frame)."""
    for offset in range(search):
        idx = frame_idx + offset
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        res = sp.DETECTOR.predict(frame, conf=0.15, verbose=False, device=sp.DEVICE)[0]
        if len(res.boxes) > 0:
            best = int(res.boxes.conf.argmax())
            if mode == "keypoints":
                kbox = _keypoint_box(res, best, pad, frame.shape[0], frame.shape[1])
                if kbox is not None:
                    return kbox
            box = res.boxes.xyxy[best].cpu().numpy()
            return sp.get_square_box(box, frame.shape[0], frame.shape[1], padding=pad)
    return None


def _video_path(row):
    vp_folder, file_id_str = row["file_id"].split("/")
    return os.path.join(C.DATA_ROOT, "data", row["camera"], vp_folder, f"{file_id_str}.mp4")


def _read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ret, frame = cap.read()
    return frame if ret else None


def _stratified(manifest, n, seed):
    """~n rows spread evenly across obj_class."""
    classes = list(manifest["obj_class"].unique())
    per = max(1, n // len(classes))
    parts = [g.sample(min(per, len(g)), random_state=seed)
             for _, g in manifest.groupby("obj_class")]
    out = pd.concat(parts).sort_values("obj_class").reset_index(drop=True)
    return out.head(n)


def preview(manifest, n, pad, seed, mode):
    out_dir = os.path.join(C.PKG_DIR, "viz", "roi_preview")
    os.makedirs(out_dir, exist_ok=True)
    rows = _stratified(manifest, n, seed)
    dims, fallbacks, made = [], 0, 0

    for _, row in rows.iterrows():
        vpath = _video_path(row)
        if not os.path.exists(vpath):
            continue
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            cap.release(); continue
        frame = _read_frame(cap, row["frame"])
        if frame is None:
            cap.release(); continue
        H, W = frame.shape[:2]
        box = roi_box_for_frame(cap, int(row["frame"]), pad, mode)
        cap.release()

        roi = sp.crop_roi(frame, box)
        if box is None:
            fallbacks += 1
        else:
            x1, y1, x2, y2 = box
            dims.append((x2 - x1, y2 - y1, W, H))

        # side-by-side: full frame with ROI rectangle | cropped ROI
        full_vis = frame.copy()
        if box is not None:
            cv2.rectangle(full_vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
        target_h = 360
        full_r = cv2.resize(full_vis, (int(W * target_h / H), target_h))
        rh, rw = roi.shape[:2]
        roi_r = cv2.resize(roi, (int(rw * target_h / rh), target_h))
        gap = np.full((target_h, 12, 3), 255, np.uint8)
        combo = np.hstack([full_r, gap, roi_r])
        tag = "full-frame ROI box  |  cropped ROI" + ("  (FALLBACK: full frame)" if box is None else "")
        cv2.putText(combo, f"{row['obj_class']}  {row['split']}  {tag}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        name = f"{row['obj_class']}__{os.path.splitext(row['image'])[0]}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), combo, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        made += 1

    print(f"\nPreview: {made} side-by-side images -> {out_dir}")
    if dims:
        d = np.array(dims, float)
        w_frac, h_frac = d[:, 0] / d[:, 2], d[:, 1] / d[:, 3]
        area_frac = (d[:, 0] * d[:, 1]) / (d[:, 2] * d[:, 3])
        print(f"\nROI dimension stats (pad={pad}, source frame {int(d[0,2])}x{int(d[0,3])}):")
        print(f"  ROI side px      min/med/max : {d[:,0].min():.0f} / {np.median(d[:,0]):.0f} / {d[:,0].max():.0f}")
        print(f"  ROI width frac   min/med/max : {w_frac.min():.2f} / {np.median(w_frac):.2f} / {w_frac.max():.2f}")
        print(f"  ROI height frac  min/med/max : {h_frac.min():.2f} / {np.median(h_frac):.2f} / {h_frac.max():.2f}")
        print(f"  ROI area frac    min/med/max : {area_frac.min():.2f} / {np.median(area_frac):.2f} / {area_frac.max():.2f}")
    print(f"  ROI mode: {mode}")
    print(f"  fell back to full frame (no pose detection): {fallbacks}/{made}")


def extract_all(manifest, pad, out_dir, mode="keypoints", keep_fallback=False):
    """Extract a padded ROI per manifest row.

    Frames with NO pose detection (crop_roi would fall back to the full frame) are
    DROPPED by default — keeping a full frame in a ROI dataset reintroduces the
    tiny-object problem and mixes scales. They are logged to fallbacks.csv so the
    count is transparent. Pass keep_fallback=True to save them as full frames."""
    os.makedirs(out_dir, exist_ok=True)
    written, missing, dropped = 0, 0, []
    bad = set()
    for _, row in manifest.iterrows():
        vpath = _video_path(row)
        if vpath in bad or not os.path.exists(vpath):
            missing += 1; continue
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            bad.add(vpath); cap.release(); missing += 1; continue
        frame = _read_frame(cap, row["frame"])
        if frame is None:
            cap.release(); missing += 1; continue
        box = roi_box_for_frame(cap, int(row["frame"]), pad, mode)
        cap.release()
        if box is None and not keep_fallback:
            dropped.append(row)            # no driver found -> exclude from ROI set
            continue
        roi = sp.crop_roi(frame, box)      # crop_roi returns full frame if box is None
        if cv2.imwrite(os.path.join(out_dir, row["image"]), roi, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            written += 1
        else:
            missing += 1

    msg = f"Extracted {written} ROI crops to {out_dir}"
    if missing:
        msg += f"  ({missing} missing videos/frames)"
    if dropped:
        log = os.path.join(out_dir, "fallbacks.csv")
        pd.DataFrame(dropped).to_csv(log, index=False)
        by = pd.DataFrame(dropped)["obj_class"].value_counts().to_dict()
        msg += f"\nDropped {len(dropped)} no-pose-detection frames (logged -> {log}): {by}"
    elif keep_fallback:
        msg += "  (--keep-fallback: no-detection frames saved as full frames)"
    print(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=C.FROZEN_MANIFEST,
                    help="frozen manifest to extract (default: config.FROZEN_MANIFEST)")
    ap.add_argument("--roi", choices=["keypoints", "person"], default="keypoints",
                    help="ROI definition: keypoints = face+hands region (object-focused, tighter); "
                         "person = full person box (spatial-pipeline behavior)")
    ap.add_argument("--pad", type=float, default=0.40,
                    help="square-ROI padding around the keypoint/person box")
    ap.add_argument("--preview", type=int, default=None, metavar="N",
                    help="only build N side-by-side preview images + ROI stats")
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--out", default=C.SAMPLED_DIR,
                    help="output dir for ROI crops (default: config.SAMPLED_DIR, read by stage 2)")
    ap.add_argument("--keep-fallback", action="store_true",
                    help="save no-pose-detection frames as full frames instead of dropping them")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    print(f"Manifest: {len(manifest)} frames  ({os.path.basename(args.manifest)})")
    if args.preview:
        preview(manifest, args.preview, args.pad, args.seed, args.roi)
    else:
        extract_all(manifest, args.pad, args.out, mode=args.roi, keep_fallback=args.keep_fallback)


if __name__ == "__main__":
    main()
