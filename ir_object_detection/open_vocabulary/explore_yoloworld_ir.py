#!/usr/bin/env python3
"""
Open-vocabulary live probe — IR security camera (RTSP)

Exploration tool, NOT inference.  This does not run the DriveGuard
spatio-temporal model at all.  It points an open-vocabulary detector
(YOLO-World or YOLOE) at the live IR stream and visualizes the *raw* objects
array it produces every frame — every box, label and confidence — plus a
console dump.

Purpose: empirically answer the only question that decides whether an
object-level Phone verifier is worth building — does a modern open-vocab
detector actually see phones (and at what confidence) on the new Reolink IR
camera, where the old COCO detector fired on only ~24% of clips?

Stream handling (background freshest-frame grabber, downscale, auto-reconnect)
is copied from infer_live_ir.py so this file is fully self-contained.

Usage:
  # Default IR cam, trimmed winning prompts:
  python explore_yoloworld_ir.py

  # Label a run and log per-frame confidences to CSV for discrimination analysis:
  python explore_yoloworld_ir.py --label phone --log phone_run.csv
  python explore_yoloworld_ir.py --label safe  --log safe_run.csv

  # Custom prompts / lower threshold:
  python explore_yoloworld_ir.py \\
      --classes "cell phone,smartphone,bottle,mug" --conf 0.03

  # YOLO-World instead of the YOLOE default, record the annotated stream:
  python explore_yoloworld_ir.py --model yolov8x-worldv2.pt --record ./yw_probe.mp4

  # Lighter/faster model:
  python explore_yoloworld_ir.py --model yolov8s-worldv2.pt
"""

import argparse
import csv
import os
import sys
import threading
import time
from pathlib import Path as _Path
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(_Path(__file__).parent.parent.parent / 'DriveGuard' / '.env')
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE, YOLOWorld

# ──────────────────────────────────────────────────────────────────────────────
# Stream config (matches infer_live_ir.py)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_RTSP = os.environ.get('RTSP_URL', 'rtsp://<user>:<pass>@<host>:554/h264Preview_01_main')

os.environ.setdefault(
    'OPENCV_FFMPEG_CAPTURE_OPTIONS',
    'rtsp_transport;tcp|max_delay;500000',
)

PROC_WIDTH        = 1000
FRAME_SKIP        = 3
RECONNECT_DELAY   = 2.0
MAX_READ_FAILURES = 30

# Trimmed to empirically proven winners from IR probe runs.
# Dropped: mobile phone, phone in hand (low conf/rate), water bottle, plastic bottle,
#          soda can, can (redundant with bottle), coffee cup, paper cup (redundant with mug/cup).
# person/face dropped — caused phantom detections on empty frames.
DEFAULT_CLASSES = [
    # ── Phone ─────────────────────────────────────────────────────────────────
    'smartphone',        # mean conf 0.57 — cleanest phone signal
    'cell phone',        # mean conf 0.29 — broader coverage
    # ── Drink: bottles ────────────────────────────────────────────────────────
    'bottle',            # mean conf 0.32
    'bottle in hand',    # relational phrasing — to be evaluated
    'drinking water',    # substance phrasing — to be evaluated
    'wine bottle',       # specific shape — to be evaluated
    # ── Drink: cups / mugs ────────────────────────────────────────────────────
    'mug',               # mean conf 0.67 — strongest cup signal
    'cup',               # mean conf 0.61
    'drinking glass',    # mean conf 0.27
    # ── Food ──────────────────────────────────────────────────────────────────
    'apple',             # mean conf 0.32 — only food with signal so far
    'sandwich',
    'burger',
    'banana',
    'donut',
    'snack',
    'food',
]

# Distinct BGR colors cycled per class index for the boxes.
_PALETTE = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 200, 0),
    (255, 128, 0), (255, 0, 0), (255, 0, 255), (128, 0, 255),
    (200, 200, 0), (0, 128, 255),
]

# ──────────────────────────────────────────────────────────────────────────────
# Stream helpers (copied from infer_live_ir.py)
# ──────────────────────────────────────────────────────────────────────────────

def open_stream(source: str) -> cv2.VideoCapture:
    """Open an RTSP/file/device source via FFMPEG with a small buffer."""
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


class FreshestFrame(threading.Thread):
    """Background grabber that always holds only the latest decoded frame."""

    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True)
        self.cap      = cap
        self.lock     = threading.Lock()
        self.frame    = None
        self.failures = 0
        self.running  = True
        self.start()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.failures += 1
                time.sleep(0.01)
                continue
            self.failures = 0
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame

    def stop(self):
        self.running = False
        self.join(timeout=1.0)
        self.cap.release()


def downscale(frame: np.ndarray, target_w: int) -> np.ndarray:
    """Resize to target width preserving aspect ratio. No-op if already smaller."""
    h, w = frame.shape[:2]
    if target_w <= 0 or w <= target_w:
        return frame.copy()
    scale = target_w / w
    return cv2.resize(frame, (target_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)


# ──────────────────────────────────────────────────────────────────────────────
# Detection parsing + drawing
# ──────────────────────────────────────────────────────────────────────────────

def parse_detections(result, names) -> list:
    """Turn one ultralytics Result into a list of dicts: name, conf, xyxy, cls."""
    dets = []
    boxes = getattr(result, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return dets
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls  = boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        dets.append({
            'name': names[k],
            'cls' : int(k),
            'conf': float(c),
            'xyxy': (int(x1), int(y1), int(x2), int(y2)),
        })
    dets.sort(key=lambda d: d['conf'], reverse=True)
    return dets


def _box_area(xyxy) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def suppress_contained(dets: list, contain_thresh: float = 0.8) -> list:
    """Drop boxes that are mostly swallowed by a higher-confidence box.

    Plain IoU-based NMS misses a small box nested inside a much larger one
    (union is dominated by the big box, so IoU stays low even though the
    small box is almost entirely covered). This checks intersection over the
    *smaller* box's own area instead, which catches that case regardless of
    class.
    """
    kept = []
    for d in dets:  # already sorted by confidence, descending
        area = _box_area(d['xyxy'])
        if area <= 0:
            continue
        swallowed = any(
            _intersection(d['xyxy'], k['xyxy']) / area > contain_thresh
            for k in kept
        )
        if not swallowed:
            kept.append(d)
    return kept


def draw_detections(frame: np.ndarray, dets: list, live_fps: float, model_name: str):
    """Draw every detection box + a sorted side panel listing the raw array."""
    h, w = frame.shape[:2]

    for d in dets:
        x1, y1, x2, y2 = d['xyxy']
        color = _PALETTE[d['cls'] % len(_PALETTE)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d['name']} {d['conf'] * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # ── Side panel: the raw objects array, highest conf first ────────────────
    panel_w = 250
    panel_h = 28 + max(len(dets), 1) * 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f'{model_name}  {len(dets)} obj  {live_fps:.1f}fps',
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    if not dets:
        cv2.putText(frame, '(nothing detected)', (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    for i, d in enumerate(dets):
        color = _PALETTE[d['cls'] % len(_PALETTE)]
        cv2.putText(frame, f"{d['name']:<14} {d['conf'] * 100:4.0f}%",
                    (8, 44 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def run(source, model_name, classes, conf, iou, contain_thresh, proc_width, frame_skip, record_path,
        show, print_every, log_path, label):
    model_cls = YOLOE if 'yoloe' in model_name.lower() else YOLOWorld
    print(f"Loading {model_cls.__name__} model: {model_name} …")
    model = model_cls(model_name)
    model.set_classes(classes)
    print(f"Open-vocabulary prompts ({len(classes)}): {classes}")
    print(f"Confidence threshold   : {conf}")
    if log_path:
        print(f"CSV log                : {log_path}  (label={label!r})")
    print()

    print(f"Source : {source}")
    print("Connecting …", flush=True)
    cap = open_stream(source)
    if not cap.isOpened():
        print(f"Could not open stream: {source}")
        sys.exit(1)
    reader = FreshestFrame(cap)

    t_wait = time.perf_counter()
    frame = None
    while frame is None:
        ok, raw = reader.read()
        if ok:
            frame = downscale(raw, proc_width)
        elif time.perf_counter() - t_wait > 15.0:
            reader.stop()
            print(f"No frames received within 15s from: {source}")
            sys.exit(1)
        else:
            time.sleep(0.05)

    src_h, src_w = raw.shape[:2]
    fh, fw = frame.shape[:2]
    print(f"Connected.  Source: {src_w}x{src_h}  →  processing at: {fw}x{fh}")
    print("Press 'q' in the window (or Ctrl-C) to quit.\n", flush=True)

    vid_writer = None
    if record_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_writer = cv2.VideoWriter(record_path, fourcc, 25.0, (fw, fh))
        print(f"Recording annotated stream → {record_path}\n", flush=True)

    # ── CSV log setup ─────────────────────────────────────────────────────────
    # Columns: frame, timestamp_s, label, n_detections, then per-class max conf.
    # One row per frame; max conf = 0.0 when the class wasn't detected that frame.
    csv_file, csv_writer = None, None
    if log_path:
        csv_file = open(log_path, 'w', newline='')
        fieldnames = ['frame', 'timestamp_s', 'label', 'n_detections'] + classes
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    win = f'{model_cls.__name__} probe — IR'
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, fw, fh)

    seen        = Counter()
    conf_sum    = Counter()
    frames_done = 0
    dets        = []

    fps_t0, fps_frames, live_fps = time.perf_counter(), 0, 0.0
    t_start = time.perf_counter()

    try:
        while True:
            ok, raw = reader.read()
            if not ok or reader.failures >= MAX_READ_FAILURES:
                if reader.failures >= MAX_READ_FAILURES:
                    print(f"\nStream stalled ({reader.failures} fails) — reconnecting …", flush=True)
                    reader.stop()
                    time.sleep(RECONNECT_DELAY)
                    cap = open_stream(source)
                    reader = FreshestFrame(cap)
                else:
                    time.sleep(0.02)
                continue

            frame     = downscale(raw, proc_width)
            timestamp = time.perf_counter() - t_start

            if frame_skip <= 1 or frames_done % frame_skip == 0:
                results = model.predict(frame, conf=conf, iou=iou, agnostic_nms=True, verbose=False)
                dets    = parse_detections(results[0], results[0].names)
                dets    = suppress_contained(dets, contain_thresh)

                for d in dets:
                    seen[d['name']]     += 1
                    conf_sum[d['name']] += d['conf']

            fps_frames += 1
            if fps_frames >= 15:
                now = time.perf_counter()
                live_fps = fps_frames / (now - fps_t0)
                fps_t0, fps_frames = now, 0

            if print_every and frames_done % print_every == 0:
                arr = [f"{d['name']}:{d['conf']:.2f}" for d in dets]
                print(f"[{frames_done:6d}] {len(dets)} objs  {arr}", flush=True)

            # ── Write one CSV row: per-class max conf this frame ──────────────
            if csv_writer is not None:
                max_conf = {c: 0.0 for c in classes}
                for d in dets:
                    if d['name'] in max_conf:
                        max_conf[d['name']] = max(max_conf[d['name']], d['conf'])
                row = {
                    'frame'         : frames_done,
                    'timestamp_s'   : f'{timestamp:.3f}',
                    'label'         : label,
                    'n_detections'  : len(dets),
                }
                row.update(max_conf)
                csv_writer.writerow(row)

            draw_detections(frame, dets, live_fps, model_cls.__name__)

            if vid_writer is not None:
                vid_writer.write(frame)
            if show:
                cv2.imshow(win, frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    print("\nQuit requested.", flush=True)
                    break

            frames_done += 1

    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        reader.stop()
        if vid_writer is not None:
            vid_writer.release()
        if csv_file is not None:
            csv_file.close()
            print(f"CSV log saved: {log_path}")
        if show:
            cv2.destroyAllWindows()

    # ── Session summary: detection rate + mean conf per prompt ───────────────
    total   = max(frames_done, 1)
    elapsed = time.perf_counter() - t_start
    print("\n" + "═" * 52)
    print(f"  {model_cls.__name__} probe summary  ({frames_done} frames, {elapsed:.1f}s)")
    if label:
        print(f"  Label: {label}")
    print("─" * 52)
    print(f"  {'class':<18}{'boxes':>8}{'rate':>9}{'mean conf':>12}")
    if seen:
        for name in sorted(seen, key=lambda n: seen[n], reverse=True):
            n = seen[name]
            print(f"  {name:<18}{n:>8}{n / total * 100:>8.1f}%{conf_sum[name] / n:>11.2f}")
    else:
        print("  (no detections all session)")
    print("═" * 52)
    if record_path:
        print(f"Recording saved: {record_path}")
    print("Done.")


def main():
    p = argparse.ArgumentParser(
        description='Live open-vocabulary (YOLO-World / YOLOE) probe on the IR camera (visualization only).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--source', default=DEFAULT_RTSP,
                   help=f'RTSP URL / video file / device index (default: IR cam)')
    p.add_argument('--model', default='yoloe-26l-seg.pt',
                   help='YOLO-World weights (yolov8{s,m,l,x}-world.pt or -worldv2.pt) or '
                        'YOLOE weights (yoloe-26{n,s,m,l,x}-seg.pt, yoloe-11{s,m,l}-seg.pt); '
                        'auto-downloaded. Default: yoloe-26l-seg.pt')
    p.add_argument('--classes', default=None,
                   help='Comma-separated open-vocabulary prompts. '
                        f'Default: {",".join(DEFAULT_CLASSES)}')
    p.add_argument('--conf', type=float, default=0.05,
                   help='Confidence threshold (low by default to surface weak IR detections)')
    p.add_argument('--iou', type=float, default=0.5,
                   help='NMS IoU threshold, applied class-agnostically so overlapping prompts '
                        '(e.g. "smartphone" vs "cell phone") on the same object collapse to one '
                        'box (default 0.5)')
    p.add_argument('--contain_thresh', type=float, default=0.8,
                   help='Drop a box if this fraction of its own area is covered by a higher-'
                        'confidence box (catches small-box-inside-big-box duplicates that plain '
                        'IoU NMS misses; default 0.8, 1.0 = disable)')
    p.add_argument('--proc_width', type=int, default=PROC_WIDTH,
                   help=f'Downscale width before detection (default {PROC_WIDTH}; 0 = full res)')
    p.add_argument('--frame_skip', type=int, default=FRAME_SKIP,
                   help=f'Run detection every Nth frame, reusing the last result in between '
                        f'(default {FRAME_SKIP}; 1 = every frame)')
    p.add_argument('--record', default=None, help='Optional path to save annotated stream as mp4')
    p.add_argument('--no_show', action='store_true', help='Headless (use with --record)')
    p.add_argument('--print_every', type=int, default=15,
                   help='Print the raw detections array every N frames (0 = never)')
    p.add_argument('--log', default=None, metavar='PATH',
                   help='Write per-frame per-class max-confidence to this CSV file')
    p.add_argument('--label', default='',
                   help='Ground-truth tag written to the label column in --log '
                        '(e.g. "phone", "safe", "bottle")')
    args = p.parse_args()

    classes = [c.strip() for c in args.classes.split(',')] if args.classes else DEFAULT_CLASSES

    run(
        source         = args.source,
        model_name     = args.model,
        classes        = classes,
        conf           = args.conf,
        iou            = args.iou,
        contain_thresh = args.contain_thresh,
        proc_width     = args.proc_width,
        frame_skip     = args.frame_skip,
        record_path    = args.record,
        show           = not args.no_show,
        print_every    = args.print_every,
        log_path       = args.log,
        label          = args.label,
    )


if __name__ == '__main__':
    main()
