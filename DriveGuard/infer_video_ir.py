#!/usr/bin/env python3
"""
DriveGuard — offline video inference with YOLOE-26-L object-detection fusion.

Same pipeline as DriveGuard/infer_live_ir.py, but the frame source is a file
instead of an RTSP stream, and the result is written to an annotated video
rather than shown in the desktop GUI.

Every model, constant and decision function is imported from infer_live_ir --
nothing is reimplemented here -- so this script and the live app cannot drift
apart. The only differences are structural, and both follow from the source
being a file:

  * No FreshestFrame. The live app drops stale frames to stay current with a
    camera; a file has no such deadline, so every frame is read in order and
    nothing is skipped. Inference still runs every STEP-th frame, exactly as
    live, so the sampled sequence fed to the temporal model is identical.
  * The class panel is burned into the output video. In the live app those
    rows are drawn by the CustomTkinter UI, which does not exist here.

Usage
-----
    python infer_video_ir.py --video clip.mp4
    python infer_video_ir.py --video clip.mp4 --output out.mp4 --show
    python infer_video_ir.py --video clip.mp4 --no-fusion        # temporal only
    python infer_video_ir.py --video clip.mp4 --fusion-weight 0.6
"""
import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer_live_ir as m


# ──────────────────────────────────────────────────────────────────────────────
# Overlay
# ──────────────────────────────────────────────────────────────────────────────

PANEL_W = 250
FONT    = cv2.FONT_HERSHEY_SIMPLEX


def _blend_panel(frame, x, y, w, h, alpha=0.65):
    """Darken a rectangle in place so text stays legible over bright IR."""
    roi = frame[y:y + h, x:x + w]
    if roi.size:
        roi[:] = cv2.addWeighted(roi, 1 - alpha, roi * 0 + 20, alpha, 0)


def draw_panel(frame, cls_idx, temporal_probs, prior, fused_probs,
               initialized, n_features, obj_token, t_sec):
    """Burn the live app's ClassPanel (Temporal / OD Prior / Fused) into a frame."""
    h, w = frame.shape[:2]
    x0 = w - PANEL_W
    _blend_panel(frame, x0, 0, PANEL_W, h)

    if not initialized:
        cv2.putText(frame, 'Initializing', (x0 + 14, 34), FONT, 0.62,
                    (170, 170, 170), 2, cv2.LINE_AA)
        cv2.putText(frame, f'{n_features}/{m.WINDOW_FRAMES} features',
                    (x0 + 14, 60), FONT, 0.44, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(frame, f't={t_sec:.1f}s', (x0 + 14, 82), FONT, 0.42,
                    (120, 120, 120), 1, cv2.LINE_AA)
        return frame

    probs = fused_probs if fused_probs is not None else temporal_probs
    colour = m.CLASS_COLORS[cls_idx]

    cv2.putText(frame, m.CLASSES[cls_idx], (x0 + 14, 40), FONT, 0.95,
                colour, 2, cv2.LINE_AA)
    cv2.putText(frame, f'{probs[cls_idx] * 100:.1f}%', (x0 + 14, 68), FONT,
                0.55, colour, 1, cv2.LINE_AA)

    # Per-class bars for whichever distribution is driving the decision
    bar_x, bar_w, bar_h = x0 + 14, PANEL_W - 76, 9
    for i, name in enumerate(m.CLASSES):
        by = 92 + i * 26
        cv2.rectangle(frame, (bar_x, by), (bar_x + bar_w, by + bar_h),
                      (55, 55, 55), -1)
        cv2.rectangle(frame, (bar_x, by),
                      (bar_x + int(bar_w * float(probs[i])), by + bar_h),
                      m.CLASS_COLORS[i], -1)
        cv2.putText(frame, name, (bar_x, by - 3), FONT, 0.38,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f'{float(probs[i]) * 100:3.0f}%',
                    (bar_x + bar_w + 6, by + bar_h), FONT, 0.38,
                    (200, 200, 200), 1, cv2.LINE_AA)

    # The three fusion rows, mirroring the GUI
    def _row(label, vec, y, dim=False):
        col = (150, 150, 150) if dim else (225, 225, 225)
        if vec is None:
            cv2.putText(frame, f'{label}: --', (x0 + 14, y), FONT, 0.40,
                        (110, 110, 110), 1, cv2.LINE_AA)
            return
        txt = '  '.join(f'{n[0]} {float(v) * 100:.0f}%'
                        for n, v in zip(m.CLASSES, vec))
        cv2.putText(frame, f'{label}: {txt}', (x0 + 14, y), FONT, 0.40,
                    col, 1, cv2.LINE_AA)

    _row('Temporal', temporal_probs, 190, dim=True)
    _row('OD Prior', prior,          210, dim=True)
    _row('Fused',    fused_probs,    230)

    cv2.putText(frame, f'obj: {obj_token or "none"}', (x0 + 14, 258), FONT,
                0.40, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(frame, f't={t_sec:.1f}s', (x0 + 14, 278), FONT, 0.40,
                (140, 140, 140), 1, cv2.LINE_AA)
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Video inference
# ──────────────────────────────────────────────────────────────────────────────

def run_video(video: str,
              output: str = 'driveguard_annotated.mp4',
              proc_width: int = m.PROC_WIDTH,
              models: dict = None,
              enable_fusion: bool = True,
              fusion_weight: float = m.FUSION_WEIGHT,
              spatial_weights: str = None,
              temporal_weights: str = None,
              show: bool = False):
    """Run the live pipeline over a video file and write an annotated copy."""
    if models is None:
        # build_models() expects resolved paths — the GUI resolves them the same
        # way before calling it (app_ir_gui.py:404-406).
        spatial_weights  = spatial_weights  or m._ensure_downloaded('vit_spatial_model_v1.pth')
        temporal_weights = temporal_weights or m._ensure_downloaded('temporal_head_model.pth')
        models = m.build_models(spatial_weights, temporal_weights)
    device, use_fp16 = models['device'], models['use_fp16']
    yolo           = models['yolo']
    yolo_obj       = models.get('yolo_obj') if enable_fusion else None
    spatial_model  = models['spatial']
    temporal_model = models['temporal']

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"Could not open video: {video}")
        return None

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video   : {video}")
    print(f"Source  : {src_w}x{src_h}  {src_fps:.2f} fps"
          + (f"  {n_frames} frames  ({n_frames / src_fps:.1f}s)" if n_frames else ""))
    print(f"Fusion  : {'ON  (W=' + str(fusion_weight) + ')' if yolo_obj else 'OFF (temporal only)'}")

    ok, raw = cap.read()
    if not ok:
        print("Video contains no readable frames.")
        cap.release()
        return None
    frame = m.downscale(raw, proc_width)
    img_h, img_w = frame.shape[:2]
    print(f"Process : {img_w}x{img_h}  →  {output}\n")

    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'),
                             src_fps, (img_w, img_h))

    roi            = (0, 0, img_w, img_h)
    feature_deque  = deque(maxlen=m.WINDOW_FRAMES)
    object_deque   = deque(maxlen=m.WINDOW_FRAMES)
    temporal_probs = prior = fused_probs = None
    current_cls    = m.CLASSES.index('Safe')
    current_obj    = None
    initialized    = False
    predictions    = []
    fidx           = 0
    t0             = time.perf_counter()

    try:
        while True:
            if fidx > 0:
                ok, raw = cap.read()
                if not ok:
                    break
                frame = m.downscale(raw, proc_width)

            # ROI refresh — identical cadence to the live loop
            if fidx % m.CYCLE_SIZE == 0:
                new_roi = m.detect_roi_from_frame(frame, yolo, img_h, img_w)
                if new_roi is not None:
                    roi = new_roi
                if fidx == 0:
                    print(f"Initial ROI : {roi}", flush=True)

            if fidx % m.STEP == 0:
                batch = m.crop_and_preprocess(frame, roi).unsqueeze(0).to(device)
                if use_fp16:
                    batch = batch.half()
                with torch.no_grad():
                    feature_deque.append(spatial_model(batch).squeeze(0))

                if yolo_obj is not None:
                    kps       = m.get_person_keypoints(frame, yolo)
                    kp_radius = m.OBJ_KP_RADIUS_FRAC * max(roi[2] - roi[0],
                                                           roi[3] - roi[1])
                    tok, box, conf = m.detect_object_token(
                        frame, yolo_obj, roi, kps, kp_radius)
                    object_deque.append((tok, conf))
                    disp = m.persistent_token(object_deque)
                    current_obj = disp if (disp is not None and tok == disp) else None

                if len(feature_deque) == m.WINDOW_FRAMES:
                    seq = torch.stack(list(feature_deque)).unsqueeze(0)
                    with torch.no_grad():
                        logits = temporal_model(seq)
                    temporal_probs = torch.softmax(logits, -1).float().cpu().squeeze(0)

                    prior = fused_probs = None
                    if yolo_obj is not None and len(object_deque) == m.WINDOW_FRAMES:
                        prior       = m.compute_object_prior(object_deque)
                        fused_probs = m.fuse_probs(temporal_probs, prior, fusion_weight)

                    probs       = fused_probs if fused_probs is not None else temporal_probs
                    current_cls = int(probs.argmax())
                    predictions.append(m.CLASSES[current_cls])
                    if not initialized:
                        initialized = True
                        print(f"  Initialized at frame {fidx} "
                              f"(first: {m.CLASSES[current_cls]})", flush=True)

            roi_colour = m.CLASS_COLORS[current_cls] if initialized else (160, 160, 160)
            m.draw_roi(frame, roi, roi_colour)
            draw_panel(frame, current_cls, temporal_probs, prior, fused_probs,
                       initialized, len(feature_deque), current_obj,
                       fidx / src_fps)

            writer.write(frame)
            if show:
                cv2.imshow('DriveGuard — video', frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            fidx += 1
            if n_frames and fidx % 150 == 0:
                print(f"  {fidx}/{n_frames} frames "
                      f"({100 * fidx / n_frames:.0f}%)", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted — writing what has been processed so far.")
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────────
    wall = time.perf_counter() - t0
    counts = Counter(predictions)
    print(f"\nWrote {output}")
    print(f"Processed {fidx} frames in {wall:.1f}s "
          f"({fidx / wall:.1f} fps, {fidx / src_fps / wall:.2f}x real-time)")
    if predictions:
        print(f"\nWindows classified: {len(predictions)}")
        for name in m.CLASSES:
            n = counts.get(name, 0)
            print(f"  {name:6} {n:5}  ({100 * n / len(predictions):5.1f}%)")
        print(f"\nMajority: {counts.most_common(1)[0][0]}")
    else:
        print("\nNo full 16-frame window was collected — clip too short "
              f"(needs ≥ {m.WINDOW_FRAMES * m.STEP} frames).")
    return counts


def main():
    ap = argparse.ArgumentParser(
        description='DriveGuard offline video inference (YOLOE fusion pipeline).')
    ap.add_argument('--video', required=True, help='input video file')
    ap.add_argument('--output', default='driveguard_annotated.mp4',
                    help='annotated output video (default: %(default)s)')
    ap.add_argument('--no-fusion', action='store_true',
                    help='disable YOLOE object-detection fusion (temporal only)')
    ap.add_argument('--fusion-weight', type=float, default=m.FUSION_WEIGHT,
                    help='W in log-linear fusion (default: %(default)s)')
    ap.add_argument('--proc-width', type=int, default=m.PROC_WIDTH,
                    help='processing width (default: %(default)s)')
    ap.add_argument('--spatial_weights', default=None)
    ap.add_argument('--temporal_weights', default=None)
    ap.add_argument('--show', action='store_true',
                    help='display frames while processing (press q to stop)')
    a = ap.parse_args()

    run_video(video=a.video, output=a.output, proc_width=a.proc_width,
              enable_fusion=not a.no_fusion, fusion_weight=a.fusion_weight,
              spatial_weights=a.spatial_weights,
              temporal_weights=a.temporal_weights, show=a.show)


if __name__ == '__main__':
    main()
