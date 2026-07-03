#!/usr/bin/env python3
"""
DriveGuard Live Inference — refactored for GUI integration.

Same pipeline as ../infer_live_ir.py with two additions:
  - build_models()  : load all models upfront; returns a dict so the GUI can
                      cache them and avoid reloading on Stop → Start.
  - run_live()      : extended with headless / callback params so the GUI can
                      display frames and predictions without cv2.imshow.

Weights are resolved from <repo>/pipelines/vit_transformer/ where they live.
"""

import os
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

# Weights live at <repo>/pipelines/vit_transformer/
_WEIGHTS_DIR = Path(__file__).parent.parent / 'pipelines' / 'vit_transformer'

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
    path = _WEIGHTS_DIR / filename
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

CLASSES      = ['Drink', 'Phone', 'Safe']
CLASS_COLORS = {
    0: (0,   165, 255),
    1: (0,   0,   255),
    2: (0,   200, 0  ),
}

WINDOW_FRAMES     = 16
STEP              = 6
CYCLE_SIZE        = 90
IMG_SIZE          = 384
ROI_PADDING       = 0.08
YOLO_CONF         = 0.25
RECONNECT_DELAY   = 2.0
MAX_READ_FAILURES = 30
PROC_WIDTH        = 640

NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD  = [0.5, 0.5, 0.5]

# ──────────────────────────────────────────────────────────────────────────────
# Temporal Model
# ──────────────────────────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    def __init__(self, cfg):
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
        return self.norm(self.transformer(x))[:, 0]


class SingleViewDriveTransformer(nn.Module):
    def __init__(self, cfg):
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
    return _PREPROCESS(Image.fromarray(
        cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)))

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
    state = {k.replace('_orig_mod.', ''): v
             for k, v in torch.load(weights_path, map_location='cpu').items()}
    model.load_state_dict(state)
    model = model.to(device)
    return (model.half() if use_fp16 else model).eval()


def build_models(spatial_weights: str, temporal_weights: str) -> dict:
    """
    Load all models and return as a dict for GUI caching.
    Call from a background thread — takes ~30s on first run.
    Returns: {'device', 'use_fp16', 'yolo', 'spatial', 'temporal'}
    """
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
    yolo = YOLO(str(_WEIGHTS_DIR / 'yolov8n-pose.pt'))
    spatial_model  = load_spatial_model(spatial_weights, device, use_fp16)
    temporal_model = load_temporal_model(temporal_weights, device, use_fp16)
    print("  Models ready.\n")

    return dict(device=device, use_fp16=use_fp16,
                yolo=yolo, spatial=spatial_model, temporal=temporal_model)

# ──────────────────────────────────────────────────────────────────────────────
# Overlay Rendering
# ──────────────────────────────────────────────────────────────────────────────

def draw_roi(frame, roi, color):
    """Draw only the ROI box — all other info lives in the app UI."""
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
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

def run_live(
    source: str,
    proc_width: int = PROC_WIDTH,
    # ── GUI params ─────────────────────────────────────────────────────────────
    models: dict = None,       # pre-built from build_models(); skips loading
    headless: bool = False,    # True → skip cv2.imshow
    on_frame=None,             # callback(bgr_ndarray) — every annotated frame
    on_prediction=None,        # callback(cls_idx, probs, initialized)
    should_run=None,           # callable() → bool; loop exits when False
    on_status=None,            # callback(text, color)
    # ── CLI-only params (ignored when models= is provided) ────────────────────
    spatial_weights: str = None,
    temporal_weights: str = None,
    record_path: str = None,
    show: bool = True,
):
    def _status(text, color='#888888'):
        if on_status:
            on_status(text, color)
        else:
            print(text)

    # ── Models ────────────────────────────────────────────────────────────────
    if models is not None:
        device, use_fp16 = models['device'], models['use_fp16']
        yolo = models['yolo']
        spatial_model  = models['spatial']
        temporal_model = models['temporal']
    else:
        if torch.backends.mps.is_available():
            device, use_fp16 = torch.device('mps'), True
        elif torch.cuda.is_available():
            device, use_fp16 = torch.device('cuda'), True
        else:
            device, use_fp16 = torch.device('cpu'), False
        print("\nLoading models …")
        yolo           = YOLO(str(_WEIGHTS_DIR / 'yolov8n-pose.pt'))
        spatial_model  = load_spatial_model(spatial_weights, device, use_fp16)
        temporal_model = load_temporal_model(temporal_weights, device, use_fp16)
        print("  Models ready.\n")

    # ── Open stream ────────────────────────────────────────────────────────────
    _status('Connecting…', '#ffaa00')
    cap = open_stream(source)
    if not cap.isOpened():
        _status(f'Could not open: {source}', '#ff4444')
        if not headless:
            sys.exit(1)
        return
    reader = FreshestFrame(cap)

    t_wait = time.perf_counter(); frame = None
    while frame is None:
        ok, raw = reader.read()
        if ok:
            frame = downscale(raw, proc_width)
        elif time.perf_counter() - t_wait > 15.0:
            reader.stop(); _status('No frames in 15s', '#ff4444')
            if not headless:
                sys.exit(1)
            return
        else:
            time.sleep(0.05)
        if should_run is not None and not should_run():
            reader.stop(); return

    src_h, src_w = raw.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    img_h, img_w = frame_h, frame_w
    print(f"Connected.  Source: {src_w}x{src_h}  →  processing: {frame_w}x{frame_h}")

    writer = None
    if record_path and not headless:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(record_path, fourcc, 25.0, (frame_w, frame_h))

    if show and not headless:
        cv2.namedWindow(f'DriveGuard — {CAMERA}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'DriveGuard — {CAMERA}', frame_w, frame_h)

    roi           = (0, 0, img_w, img_h)
    feature_deque = deque(maxlen=WINDOW_FRAMES)
    initialized   = False
    all_predictions = []
    current_cls   = CLASSES.index('Safe')
    current_probs = torch.zeros(len(CLASSES)); current_probs[current_cls] = 1.0
    fidx = 0; t_start = fps_t0 = time.perf_counter()
    fps_frames = 0; live_fps = 0.0

    try:
        while True:
            if should_run is not None and not should_run():
                break

            ok, raw = reader.read()
            if not ok or reader.failures >= MAX_READ_FAILURES:
                if reader.failures >= MAX_READ_FAILURES:
                    _status('Stream stalled — reconnecting…', '#ffaa00')
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

                # Report init progress to the UI on each new feature
                if not initialized:
                    n = len(feature_deque)
                    _status(f'Initializing…  {n} / {WINDOW_FRAMES} features', '#ffaa00')

                if len(feature_deque) == WINDOW_FRAMES:
                    seq = torch.stack(list(feature_deque)).unsqueeze(0)
                    with torch.no_grad():
                        logits = temporal_model(seq)
                    probs = torch.softmax(logits, dim=-1).float().cpu().squeeze(0)
                    current_cls   = int(probs.argmax())
                    current_probs = probs
                    all_predictions.append(CLASSES[current_cls])
                    if not initialized:
                        initialized = True
                        print(f"  Initialized  (first: {CLASSES[current_cls]})", flush=True)
                    if on_prediction is not None:
                        on_prediction(current_cls, current_probs, initialized)

            fps_frames += 1
            if fps_frames >= 30:
                now = time.perf_counter()
                live_fps = fps_frames / (now - fps_t0)
                fps_t0, fps_frames = now, 0
                if initialized:
                    _status(f'Live · {live_fps:.1f} fps', '#00cc55')

            # Only the ROI box on the video — all text/bars are in the app UI
            roi_color = CLASS_COLORS[current_cls] if initialized else (160, 160, 160)
            draw_roi(frame, roi, roi_color)

            if on_frame is not None:
                on_frame(frame)
            if writer is not None:
                writer.write(frame)
            if show and not headless:
                cv2.imshow(f'DriveGuard — {CAMERA}', frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
            fidx += 1

    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        reader.stop()
        if writer:
            writer.release()
        if show and not headless:
            cv2.destroyAllWindows()
