#!/usr/bin/env python3
"""
DriveGuard Live Inference — IR security camera (RTSP)
Single-camera driver activity recognition: Safe / Drink / Phone

Live variant of infer.py.  Instead of decoding a fixed-length video file and
writing an mp4, this consumes an unbounded RTSP stream and renders an annotated
live window (press 'q' to quit).  The full spatio-temporal pipeline is kept —
YOLO-pose ROI lock, ViT-SO400M spatial features, Transformer temporal head —
but the COCO object-detection fusion path is removed (temporal-only).

Pipeline (sliding-window with 90-frame init), identical timing to infer.py:
  Init phase  (frames 0–89):
    - Lock ROI via YOLO on frame 0
    - Sample one frame every STEP=6 frames → spatial features pushed into deque
    - Temporal model NOT run until deque has 16 features
    - Overlay shows "Initializing…"
  Streaming phase (frame 90+):
    - Each new feature → deque(maxlen=16) → temporal model runs immediately
    - ROI refreshed via YOLO once every CYCLE_SIZE=90 frames
    - Overlay shows live prediction

The stream auto-reconnects on read failure.  Weights are auto-downloaded from
Google Drive on first run (shared with infer.py).

Usage:
  python infer_live_ir.py
  python infer_live_ir.py --source "rtsp://user:pass@host:554/stream"
  python infer_live_ir.py --record ./out.mp4
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Default RTSP source (IR security camera)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_RTSP = os.environ.get('RTSP_URL', 'rtsp://<user>:<pass>@<host>:554/h264Preview_01_main')

os.environ.setdefault(
    'OPENCV_FFMPEG_CAPTURE_OPTIONS',
    'rtsp_transport;tcp|max_delay;500000',
)

# ──────────────────────────────────────────────────────────────────────────────
# Google Drive assets
# ──────────────────────────────────────────────────────────────────────────────

_DRIVE_ASSETS = {
    'vit_spatial_model_v1.pth': '13oXCmdfEl3D6088dT1gLnKY1ueZwVyHu',
    'temporal_head_model.pth' : '1lEgh8fwzBzunBQZJGhbLs-RxeJIR8zZS',
}


def _ensure_downloaded(filename: str) -> str:
    path = Path(__file__).parent / filename
    if path.exists():
        return str(path)
    file_id = _DRIVE_ASSETS[filename]
    print(f"Downloading {filename} from Google Drive …")
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
        import gdown
    gdown.download(f'https://drive.google.com/uc?id={file_id}', str(path), quiet=False)
    return str(path)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CAMERA = 'ir_sec_cam'

CLASSES       = ['Drink', 'Phone', 'Safe']
CLASS_COLORS  = {
    0: (0,   165, 255),   # Drink  — orange
    1: (0,   0,   255),   # Phone  — red
    2: (0,   200, 0  ),   # Safe   — green
}

WINDOW_FRAMES = 16
STEP          = 6
CYCLE_SIZE    = 90
IMG_SIZE      = 384
ROI_PADDING   = 0.08
YOLO_CONF     = 0.25

RECONNECT_DELAY   = 2.0
MAX_READ_FAILURES = 30
PROC_WIDTH        = 640

NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD  = [0.5, 0.5, 0.5]

# ──────────────────────────────────────────────────────────────────────────────
# Temporal Model
# ──────────────────────────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        D, H, T = cfg['input_dim'], cfg['hidden_dim'], cfg['num_frames']
        self.input_proj = nn.Linear(D, H)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, H))
        self.pos_embed  = nn.Parameter(torch.zeros(1, T + 1, H))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=cfg['num_heads'],
            dim_feedforward=cfg['dim_feedforward'], dropout=cfg['dropout'],
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg['num_layers'])
        self.norm = nn.LayerNorm(H)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.input_proj(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.norm(x)[:, 0]


class SingleViewDriveTransformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder    = TemporalEncoder(cfg)
        self.dropout    = nn.Dropout(cfg['dropout'])
        self.classifier = nn.Linear(cfg['hidden_dim'], cfg['num_classes'])

    def forward(self, x):
        return self.classifier(self.dropout(self.encoder(x)))


TEMPORAL_CFG = {
    'input_dim': 1152, 'hidden_dim': 768, 'num_heads': 8,
    'num_layers': 4, 'dim_feedforward': 2048, 'dropout': 0.3,
    'num_frames': 16, 'num_classes': 3,
}

# ──────────────────────────────────────────────────────────────────────────────
# ROI / Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def get_square_box(box, img_h, img_w, padding=ROI_PADDING):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + padding) / 2
    return (max(0, int(cx - half)), max(0, int(cy - half)),
            min(img_w, int(cx + half)), min(img_h, int(cy + half)))


def detect_roi_from_frame(frame, yolo_model, img_h, img_w):
    results = yolo_model(frame, conf=YOLO_CONF, verbose=False)
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        return get_square_box(boxes[int(np.argmax(confs))], img_h, img_w)
    return None


_PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def crop_and_preprocess(frame_bgr, roi):
    x1, y1, x2, y2 = roi
    crop = frame_bgr[y1:y2, x1:x2]
    return _PREPROCESS(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_spatial_model(weights_path, device, use_fp16):
    print(f"  Spatial  model ← {weights_path}")
    model = timm.create_model('vit_so400m_patch14_siglip_384', pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.reset_classifier(0)
    model = model.to(device)
    return (model.half() if use_fp16 else model).eval()


def load_temporal_model(weights_path, device, use_fp16, cfg=None):
    print(f"  Temporal model ← {weights_path}")
    model = SingleViewDriveTransformer(cfg or TEMPORAL_CFG)
    state = torch.load(weights_path, map_location='cpu')
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device)
    return (model.half() if use_fp16 else model).eval()

# ──────────────────────────────────────────────────────────────────────────────
# Overlay Rendering
# ──────────────────────────────────────────────────────────────────────────────

def draw_overlay(frame, cls_idx, probs, roi, elapsed_s, live_fps):
    _, w = frame.shape[:2]
    color = CLASS_COLORS[cls_idx]
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    bar_x, bar_w, bar_h, pad_y = w - 170, 140, 22, 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (230, 100), (15, 15, 15), -1)
    cv2.rectangle(overlay, (bar_x - 8, pad_y - 4),
                  (w - 4, pad_y + len(CLASSES) * (bar_h + 8) + 4), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, CLASSES[cls_idx], (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
    cv2.putText(frame, f'{probs[cls_idx].item()*100:.1f}%   t={elapsed_s:.1f}s',
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, f'LIVE  {CAMERA}  {live_fps:.1f} fps',
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1, cv2.LINE_AA)

    for i, (cname, p) in enumerate(zip(CLASSES, probs.tolist())):
        by = pad_y + i * (bar_h + 8)
        cv2.rectangle(frame, (bar_x, by), (bar_x + bar_w, by + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, by), (bar_x + int(bar_w * p), by + bar_h), CLASS_COLORS[i], -1)
        cv2.putText(frame, f'{cname}  {p*100:.0f}%', (bar_x, by + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
    return frame


def draw_init_overlay(frame, roi, elapsed_s, features_collected):
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 75), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, 'Initializing...', (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, f'features: {features_collected}/{WINDOW_FRAMES}   t={elapsed_s:.1f}s',
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1, cv2.LINE_AA)
    return frame

# ──────────────────────────────────────────────────────────────────────────────
# Stream helpers
# ──────────────────────────────────────────────────────────────────────────────

def open_stream(source):
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


class FreshestFrame(threading.Thread):
    """Background grabber — always holds only the latest decoded frame."""
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap; self.lock = threading.Lock()
        self.frame = None; self.failures = 0; self.running = True
        self.start()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.failures += 1; time.sleep(0.01); continue
            self.failures = 0
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return (False, None) if self.frame is None else (True, self.frame)

    def stop(self):
        self.running = False; self.join(timeout=1.0); self.cap.release()


def downscale(frame, target_w):
    h, w = frame.shape[:2]
    if target_w <= 0 or w <= target_w:
        return frame.copy()
    scale = target_w / w
    return cv2.resize(frame, (target_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)

# ──────────────────────────────────────────────────────────────────────────────
# Main Run Loop
# ──────────────────────────────────────────────────────────────────────────────

def run_live(source, spatial_weights, temporal_weights,
             record_path=None, show=True, proc_width=PROC_WIDTH):

    if torch.backends.mps.is_available():
        device, use_fp16 = torch.device('mps'), True
        print("Device : MPS  (Apple Silicon)  — FP16 enabled")
    elif torch.cuda.is_available():
        device, use_fp16 = torch.device('cuda'), True
        print(f"Device : CUDA  ({torch.cuda.get_device_name(0)})  — FP16 enabled")
    else:
        device, use_fp16 = torch.device('cpu'), False
        print("Device : CPU  — FP32")

    print("\nLoading models …")
    yolo           = YOLO(str(Path(__file__).parent / 'yolov8n-pose.pt'))
    spatial_model  = load_spatial_model(spatial_weights, device, use_fp16)
    temporal_model = load_temporal_model(temporal_weights, device, use_fp16)
    print("  Models ready.\n")

    print(f"Camera : {CAMERA}\nSource : {source}")
    print("Connecting …", flush=True)
    cap = open_stream(source)
    if not cap.isOpened():
        print(f"Could not open stream: {source}"); sys.exit(1)
    reader = FreshestFrame(cap)

    t_wait = time.perf_counter()
    frame = None
    while frame is None:
        ok, raw = reader.read()
        if ok:
            frame = downscale(raw, proc_width)
        elif time.perf_counter() - t_wait > 15.0:
            reader.stop(); print("No frames received within 15s"); sys.exit(1)
        else:
            time.sleep(0.05)

    src_h, src_w = raw.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    img_h, img_w = frame_h, frame_w
    print(f"Connected.  Source: {src_w}x{src_h}  →  processing: {frame_w}x{frame_h}")
    print(f"Sampling : every {STEP} frames  |  Window : {WINDOW_FRAMES}  |  "
          f"ROI refresh : every {CYCLE_SIZE} frames")
    print("Press 'q' to quit.\n", flush=True)

    writer = None
    if record_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(record_path, fourcc, 25.0, (frame_w, frame_h))
        print(f"Recording → {record_path}\n", flush=True)

    if show:
        cv2.namedWindow(f'DriveGuard — {CAMERA}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'DriveGuard — {CAMERA}', frame_w, frame_h)

    roi           = (0, 0, img_w, img_h)
    feature_deque = deque(maxlen=WINDOW_FRAMES)
    initialized   = False
    all_predictions = []
    current_cls   = CLASSES.index('Safe')
    current_probs = torch.zeros(len(CLASSES)); current_probs[current_cls] = 1.0
    fidx = 0; read_failures = 0
    t_start = fps_t0 = time.perf_counter()
    fps_frames = 0; live_fps = 0.0

    try:
        while True:
            ok, raw = reader.read()
            if not ok or reader.failures >= MAX_READ_FAILURES:
                if reader.failures >= MAX_READ_FAILURES:
                    print("\nStream stalled — reconnecting …", flush=True)
                    reader.stop(); time.sleep(RECONNECT_DELAY)
                    cap = open_stream(source); reader = FreshestFrame(cap)
                    feature_deque.clear(); initialized = False
                else:
                    time.sleep(0.02)
                continue

            frame   = downscale(raw, proc_width)
            elapsed = time.perf_counter() - t_start

            if fidx % CYCLE_SIZE == 0:
                new_roi = detect_roi_from_frame(frame, yolo, img_h, img_w)
                if new_roi is not None:
                    roi = new_roi
                if fidx == 0:
                    print(f"Initial ROI : {roi}", flush=True)

            if fidx % STEP == 0:
                batch = crop_and_preprocess(frame, roi).unsqueeze(0).to(device)
                if use_fp16:
                    batch = batch.half()
                with torch.no_grad():
                    feat = spatial_model(batch).squeeze(0)
                feature_deque.append(feat)

                if len(feature_deque) == WINDOW_FRAMES:
                    seq = torch.stack(list(feature_deque)).unsqueeze(0)
                    with torch.no_grad():
                        logits = temporal_model(seq)
                    probs = torch.softmax(logits, dim=-1).float().cpu().squeeze(0)
                    current_cls = int(probs.argmax())
                    current_probs = probs
                    all_predictions.append(CLASSES[current_cls])
                    if not initialized:
                        initialized = True
                        print(f"  Initialized  (first: {CLASSES[current_cls]})", flush=True)

            fps_frames += 1
            if fps_frames >= 30:
                now = time.perf_counter()
                live_fps = fps_frames / (now - fps_t0)
                fps_t0, fps_frames = now, 0

            if initialized:
                draw_overlay(frame, current_cls, current_probs, roi, elapsed, live_fps)
            else:
                draw_init_overlay(frame, roi, elapsed, len(feature_deque))

            if writer:
                writer.write(frame)
            if show:
                cv2.imshow(f'DriveGuard — {CAMERA}', frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    print("\nQuit.", flush=True); break
            fidx += 1

    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        reader.stop()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    total_wall = time.perf_counter() - t_start
    print("\n" + "═" * 42)
    if all_predictions:
        vote  = Counter(all_predictions)
        final = vote.most_common(1)[0][0]
        total = len(all_predictions)
        print(f"  MAJORITY : {final}  ({vote[final]}/{total})")
        print("─" * 42)
        for cls in CLASSES:
            n = vote.get(cls, 0)
            print(f"  {cls:<10}  {n:>7}  {n/total*100:>5.1f}%")
    else:
        print("  No predictions (stream too short).")
    print("─" * 42)
    print(f"  Frames : {fidx}   Time : {total_wall:.1f}s")
    print("═" * 42)

# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DriveGuard — live IR-camera inference (Safe / Drink / Phone)')
    parser.add_argument('--source', default=DEFAULT_RTSP)
    parser.add_argument('--spatial_weights',  default=None)
    parser.add_argument('--temporal_weights', default=None)
    parser.add_argument('--record',           default=None)
    parser.add_argument('--proc_width', type=int, default=PROC_WIDTH)
    parser.add_argument('--no_show', action='store_true')
    args = parser.parse_args()

    run_live(
        source           = args.source,
        spatial_weights  = args.spatial_weights  or _ensure_downloaded('vit_spatial_model_v1.pth'),
        temporal_weights = args.temporal_weights or _ensure_downloaded('temporal_head_model.pth'),
        record_path      = args.record,
        show             = not args.no_show,
        proc_width       = args.proc_width,
    )


if __name__ == '__main__':
    main()
