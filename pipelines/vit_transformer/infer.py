#!/usr/bin/env python3
"""
DriveGuard Inference Script
Single-camera driver activity recognition: Safe / Drink / Phone

Pipeline (sliding-window with 90-frame init):
  Init phase  (frames 0–89):
    - Lock ROI via YOLO on frame 0
    - Sample one frame every STEP=6 frames → spatial features pushed into deque
    - Temporal model NOT run until deque has 16 features
    - Overlay shows "Initializing…"
  Streaming phase (frame 90+):
    - Continue sampling every STEP frames
    - Each new feature → deque(maxlen=16) → temporal model runs immediately
    - ROI refreshed via YOLO once every CYCLE_SIZE=90 frames
    - Overlay shows live prediction

Weights and sample video are auto-downloaded from Google Drive on first run.

Usage:
  # Auto-download weights + sample video and run:
  python infer.py

  # Custom paths:
  python infer.py \\
      --video            /path/to/video.mp4 \\
      --spatial_weights  /path/to/vit_spatial_model_v1.pth \\
      --temporal_weights /path/to/temporal_head_model.pth \\
      --output_video     ./out.mp4
"""

import argparse
import subprocess
import sys
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
# Google Drive assets  (auto-downloaded on first run)
# ──────────────────────────────────────────────────────────────────────────────

_DRIVE_ASSETS = {
    'vit_spatial_model_v1.pth': '13oXCmdfEl3D6088dT1gLnKY1ueZwVyHu',
    'temporal_head_model.pth' : '1lEgh8fwzBzunBQZJGhbLs-RxeJIR8zZS',
    'sample_video.mp4'        : '1JhhimPGppUqlsa-Yqi9IsPWl66-nGsjl',
}


def _ensure_downloaded(filename: str) -> str:
    """Download file from Google Drive if not already present. Returns local path."""
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

CAMERA = 'a_column_co_driver'

CLASSES       = ['Drink', 'Phone', 'Safe']
CLASS_COLORS  = {                             # BGR for OpenCV
    0: (0,   165, 255),   # Drink  — orange
    1: (0,   0,   255),   # Phone  — red
    2: (0,   200, 0  ),   # Safe   — green
}

WINDOW_FRAMES = 16    # temporal model input length
STEP          = 6     # sample one frame every STEP frames
CYCLE_SIZE    = 90    # ROI refresh interval (frames)
IMG_SIZE      = 384
ROI_PADDING   = 0.08
YOLO_CONF     = 0.25

# ──────────────────────────────────────────────────────────────────────────────
# Object-detection fusion  (COCO YOLOv8 soft prior over the temporal softmax)
# ──────────────────────────────────────────────────────────────────────────────
OBJ_CONF      = 0.30   # COCO detection confidence threshold
FUSION_WEIGHT = 0.6    # W — strength of the object prior in log-linear fusion
OBJ_SMOOTHING = 0.5    # Laplace smoothing on per-class window weights
OBJ_WEIGHTS   = 'yolov8m.pt'   # COCO object detector (auto-downloaded; use yolov8s/n for real-time CPU)

# False-positive control (dark-IR cabin footage is out of COCO's domain)
OBJ_KP_RADIUS_FRAC = 0.33   # object center must be within frac*ROI-size of a hand/face keypoint
KP_CONF            = 0.30    # min keypoint confidence to use for gating
RELEVANT_KP        = {0, 1, 2, 3, 4, 7, 8, 9, 10}   # nose, eyes, ears, elbows, wrists
OBJ_PERSIST_K      = 3       # display: look at last K samples
OBJ_PERSIST_MIN    = 2       # display: draw box only if token in ≥ MIN of last K

# COCO class ids → DriveGuard class.  High-precision set (drops FP-prone food/utensils).
COCO_PHONE    = {67}                       # cell phone
COCO_DRINK    = {39, 40, 41, 48}           # bottle, wine glass, cup, sandwich
OBJ_CLASS_IDS = sorted(COCO_PHONE | COCO_DRINK)   # restrict YOLO inference

# SigLIP normalisation — must match Stage 3 feature extraction
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD  = [0.5, 0.5, 0.5]

# ──────────────────────────────────────────────────────────────────────────────
# Temporal Model  (must match training notebook exactly)
# ──────────────────────────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """
    Shared encoder for a single camera view.
    Input:  (B, T=16, D=1152)
    Output: (B, hidden_dim)  via CLS token
    """
    def __init__(self, cfg: dict):
        super().__init__()
        D = cfg['input_dim']
        H = cfg['hidden_dim']
        T = cfg['num_frames']

        self.input_proj = nn.Linear(D, H)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, H))
        self.pos_embed  = nn.Parameter(torch.zeros(1, T + 1, H))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=cfg['num_heads'],
            dim_feedforward=cfg['dim_feedforward'],
            dropout=cfg['dropout'],
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg['num_layers'])
        self.norm = nn.LayerNorm(H)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x   = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.pos_embed
        x   = self.transformer(x)
        x   = self.norm(x)
        return x[:, 0]


class SingleViewDriveTransformer(nn.Module):
    """
    TemporalEncoder + Dropout + Linear classifier.
    Input:  (B, T=16, D=1152)
    Output: (B, num_classes) logits
    Must match temporal_single_cam_stage4_driveguard.ipynb Cell 5 exactly.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder    = TemporalEncoder(cfg)
        self.dropout    = nn.Dropout(cfg['dropout'])
        self.classifier = nn.Linear(cfg['hidden_dim'], cfg['num_classes'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.encoder(x)))


TEMPORAL_CFG = {
    'input_dim'      : 1152,
    'hidden_dim'     : 768,
    'num_heads'      : 8,
    'num_layers'     : 4,
    'dim_feedforward': 2048,
    'dropout'        : 0.3,
    'num_frames'     : 16,
    'num_classes'    : 3,
}

# ──────────────────────────────────────────────────────────────────────────────
# ROI Utilities
# ──────────────────────────────────────────────────────────────────────────────

def get_square_box(box, img_h: int, img_w: int, padding: float = ROI_PADDING):
    """Convert a detection box to a padded square, clipped to image bounds."""
    x1, y1, x2, y2 = box
    cx   = (x1 + x2) / 2
    cy   = (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + padding) / 2
    nx1  = max(0,     int(cx - half))
    ny1  = max(0,     int(cy - half))
    nx2  = min(img_w, int(cx + half))
    ny2  = min(img_h, int(cy + half))
    return nx1, ny1, nx2, ny2


def detect_roi_from_frame(frame: np.ndarray, yolo_model, img_h: int, img_w: int):
    """Run YOLO on one frame; return best-confidence ROI or None."""
    results = yolo_model(frame, conf=YOLO_CONF, verbose=False)
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        best  = int(np.argmax(confs))
        return get_square_box(boxes[best], img_h, img_w)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Object-detection soft fusion
# ──────────────────────────────────────────────────────────────────────────────

def get_person_keypoints(frame: np.ndarray, pose_model) -> np.ndarray:
    """
    Hand/face keypoints (nose, eyes, ears, elbows, wrists) of the most
    confident person.  Returns (N, 2) array of (x, y) or empty (0, 2).
    """
    results = pose_model(frame, conf=YOLO_CONF, verbose=False)
    if (not results or results[0].keypoints is None
            or len(results[0].boxes) == 0):
        return np.empty((0, 2))
    best = int(np.argmax(results[0].boxes.conf.cpu().numpy()))
    kp   = results[0].keypoints.data.cpu().numpy()[best]   # (17, 3): x, y, conf
    pts  = [(x, y) for i, (x, y, c) in enumerate(kp)
            if i in RELEVANT_KP and c >= KP_CONF]
    return np.array(pts, dtype=np.float32) if pts else np.empty((0, 2))


def detect_object_token(frame: np.ndarray, yolo_obj, roi: tuple,
                        keypoints: np.ndarray = None, kp_radius: float = None):
    """
    Detect a phone / drink object near the driver's hands/face.
    Returns (token, box, conf): token is 'phone'/'drink'/None, box is the
    winning detection's (x1, y1, x2, y2) int tuple (or None), conf is its score.
    Gates: (1) center inside ROI, (2) center within kp_radius of a hand/face
    keypoint (skipped if no keypoints available).  Highest-confidence wins.
    """
    rx1, ry1, rx2, ry2 = roi
    results = yolo_obj(frame, conf=OBJ_CONF, classes=OBJ_CLASS_IDS, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None, None, 0.0

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    clses = results[0].boxes.cls.cpu().numpy().astype(int)
    use_kp = keypoints is not None and len(keypoints) > 0 and kp_radius is not None

    best_token, best_box, best_conf = None, None, -1.0
    for (bx1, by1, bx2, by2), conf, cid in zip(boxes, confs, clses):
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2          # box center
        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):    # gate 1: ROI
            continue
        if use_kp:                                          # gate 2: near hand/face
            dist = np.min(np.hypot(keypoints[:, 0] - cx, keypoints[:, 1] - cy))
            if dist > kp_radius:
                continue
        if conf <= best_conf:
            continue
        best_conf  = float(conf)
        best_token = 'phone' if cid in COCO_PHONE else 'drink'
        best_box   = (int(bx1), int(by1), int(bx2), int(by2))
    return best_token, best_box, (best_conf if best_token else 0.0)


def compute_object_prior(object_deque) -> torch.Tensor:
    """
    Aggregate (token, conf) samples over the window into a smoothed prior
    ordered [Drink, Phone, Safe].  Confidence-weighted: a detection votes its
    class by `conf` and Safe by `1-conf`, so weak (likely-FP) detections barely
    move the prior; None samples vote fully Safe.
    """
    w = torch.full((len(CLASSES),), OBJ_SMOOTHING)
    di, pi, si = CLASSES.index('Drink'), CLASSES.index('Phone'), CLASSES.index('Safe')
    for tok, conf in object_deque:
        if   tok == 'drink': w[di] += conf; w[si] += (1.0 - conf)
        elif tok == 'phone': w[pi] += conf; w[si] += (1.0 - conf)
        else:                w[si] += 1.0
    return w / w.sum()


def persistent_token(object_deque) -> str:
    """
    Display de-flicker: return the latest token only if it appears in at least
    OBJ_PERSIST_MIN of the last OBJ_PERSIST_K samples, else None.
    """
    recent = [tok for tok, _ in list(object_deque)[-OBJ_PERSIST_K:]]
    if not recent or recent[-1] is None:
        return None
    return recent[-1] if recent.count(recent[-1]) >= OBJ_PERSIST_MIN else None


def fuse_probs(temporal_probs: torch.Tensor, prior: torch.Tensor,
               weight: float, eps: float = 1e-6) -> torch.Tensor:
    """
    Log-linear (product-of-experts) fusion of the temporal softmax with the
    object prior:  softmax( log(p_temporal) + W * log(prior) ).
    weight=0 (or a uniform prior) reproduces the temporal distribution.
    """
    log_p = torch.log(temporal_probs + eps) + weight * torch.log(prior + eps)
    return torch.softmax(log_p, dim=-1)

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

_PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def crop_and_preprocess(frame_bgr: np.ndarray, roi: tuple) -> torch.Tensor:
    """Crop ROI from a BGR frame and return a preprocessed (3, 384, 384) tensor."""
    x1, y1, x2, y2 = roi
    crop = frame_bgr[y1:y2, x1:x2]
    img  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return _PREPROCESS(img)

# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_spatial_model(weights_path: str, device: torch.device, use_fp16: bool) -> nn.Module:
    """Load ViT-SO400M, strip classifier head → 1152-dim feature extractor."""
    print(f"  Spatial  model ← {weights_path}")
    model = timm.create_model('vit_so400m_patch14_siglip_384', pretrained=False, num_classes=3)
    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state)
    model.reset_classifier(0)
    model = model.to(device)
    if use_fp16:
        model = model.half()
    return model.eval()


def load_temporal_model(weights_path: str, device: torch.device, use_fp16: bool,
                        cfg: dict = None) -> nn.Module:
    """Load SingleViewDriveTransformer from checkpoint."""
    print(f"  Temporal model ← {weights_path}")
    model = SingleViewDriveTransformer(cfg or TEMPORAL_CFG)
    state = torch.load(weights_path, map_location='cpu')
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device)
    if use_fp16:
        model = model.half()
    return model.eval()

# ──────────────────────────────────────────────────────────────────────────────
# Overlay Rendering
# ──────────────────────────────────────────────────────────────────────────────

def draw_overlay(
    frame: np.ndarray,
    cls_idx: int,
    probs: torch.Tensor,
    roi: tuple,
    frame_idx: int,
    fps: float,
    obj_token: str = None,
    obj_box: tuple = None,
) -> np.ndarray:
    """Annotate a frame with class prediction, confidence bars, and ROI box."""
    _, w = frame.shape[:2]
    color    = CLASS_COLORS[cls_idx]
    cls_name = CLASSES[cls_idx]
    conf     = probs[cls_idx].item()

    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # detected object box (the evidence behind the fusion prior)
    if obj_box is not None and obj_token is not None:
        ox1, oy1, ox2, oy2 = obj_box
        oc = CLASS_COLORS[CLASSES.index('Phone')] if obj_token == 'phone' \
            else CLASS_COLORS[CLASSES.index('Drink')]
        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), oc, 2)
        cv2.putText(frame, obj_token, (ox1, max(oy1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, oc, 2, cv2.LINE_AA)

    bar_x      = w - 170
    bar_w      = 140
    bar_h      = 22
    pad_y      = 12
    banner_w, banner_h = 230, 100

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (banner_w, banner_h), (15, 15, 15), -1)
    cv2.rectangle(overlay, (bar_x - 8, pad_y - 4),
                  (w - 4, pad_y + len(CLASSES) * (bar_h + 8) + 4), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, cls_name,
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
    cv2.putText(frame, f'{conf * 100:.1f}%  t={frame_idx / fps:.1f}s',
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)

    # object-detection signal (drives the fusion prior)
    obj_color = (200, 200, 200) if obj_token is None else (
        CLASS_COLORS[CLASSES.index('Phone')] if obj_token == 'phone'
        else CLASS_COLORS[CLASSES.index('Drink')])
    cv2.putText(frame, f'obj: {obj_token or "none"}',
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, obj_color, 1, cv2.LINE_AA)

    for i, (cname, p) in enumerate(zip(CLASSES, probs.tolist())):
        by   = pad_y + i * (bar_h + 8)
        fill = int(bar_w * p)
        c    = CLASS_COLORS[i]
        cv2.rectangle(frame, (bar_x, by),        (bar_x + bar_w, by + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, by),        (bar_x + fill,  by + bar_h), c, -1)
        cv2.putText(frame, f'{cname}  {p * 100:.0f}%',
                    (bar_x, by + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)

    return frame


def draw_init_overlay(
    frame: np.ndarray,
    roi: tuple,
    frame_idx: int,
    fps: float,
    features_collected: int,
) -> np.ndarray:
    """Overlay shown during the 90-frame initialization phase."""
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 75), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, 'Initializing...',
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, f'features: {features_collected}/{WINDOW_FRAMES}  t={frame_idx / fps:.1f}s',
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1, cv2.LINE_AA)

    return frame

# ──────────────────────────────────────────────────────────────────────────────
# Main Run Loop  (sliding window with 90-frame init)
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(
    video_path: str,
    spatial_weights: str,
    temporal_weights: str,
    output_video: str = 'inference_output.mp4',
    enable_fusion: bool = True,
    fusion_weight: float = FUSION_WEIGHT,
    obj_weights: str = OBJ_WEIGHTS,
):
    # ── Device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device   = torch.device('mps')
        use_fp16 = True
        print("Device : MPS  (Apple Silicon)  — FP16 enabled")
    elif torch.cuda.is_available():
        device   = torch.device('cuda')
        use_fp16 = True
        print(f"Device : CUDA  ({torch.cuda.get_device_name(0)})  — FP16 enabled")
    else:
        device   = torch.device('cpu')
        use_fp16 = False
        print("Device : CPU  — FP32")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading models …")
    yolo           = YOLO('yolov8n-pose.pt')
    spatial_model  = load_spatial_model(spatial_weights, device, use_fp16)
    temporal_model = load_temporal_model(temporal_weights, device, use_fp16)
    yolo_obj       = None
    if enable_fusion:
        print(f"  Object detector ← {obj_weights}  (fusion weight W={fusion_weight})")
        yolo_obj = YOLO(obj_weights)
    print("  Models ready.\n")

    # ── Open video ────────────────────────────────────────────────────────────
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_h, img_w = frame_h, frame_w

    print(f"Camera : {CAMERA}")
    print(f"Video  : {Path(video_path).name}  —  {total_frames} frames @ {fps:.1f} fps  "
          f"({total_frames / fps:.1f}s)")
    print(f"Sampling : every {STEP} frames  |  Window : {WINDOW_FRAMES} features  |  "
          f"ROI refresh : every {CYCLE_SIZE} frames\n")

    # ── Video writer ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_w, frame_h))

    # ── Inference state ───────────────────────────────────────────────────────
    roi           = (0, 0, img_w, img_h)        # fallback: full frame
    feature_deque = deque(maxlen=WINDOW_FRAMES)  # sliding window of spatial features
    object_deque  = deque(maxlen=WINDOW_FRAMES)  # sliding window of object tokens
    initialized   = False                        # True once deque has WINDOW_FRAMES features
    all_predictions: list = []

    current_cls   = CLASSES.index('Safe')
    current_probs = torch.zeros(len(CLASSES))
    current_probs[current_cls] = 1.0
    current_obj     = None                        # last in-ROI object token (overlay)
    current_obj_box = None                         # last in-ROI object box (overlay)

    spatial_times:  list = []
    temporal_times: list = []
    object_times:   list = []

    # ── Streaming pass ────────────────────────────────────────────────────────
    print(f"Processing …  (writing to {output_video})", flush=True)
    t_start = time.perf_counter()

    for fidx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        # ── ROI refresh (includes frame 0 for initial lock) ───────────────────
        if fidx % CYCLE_SIZE == 0:
            new_roi = detect_roi_from_frame(frame, yolo, img_h, img_w)
            if new_roi is not None:
                roi = new_roi
            if fidx == 0:
                print(f"Initial ROI : {roi}\n")

        # ── Sample frame every STEP frames ────────────────────────────────────
        if fidx % STEP == 0:
            tensor = crop_and_preprocess(frame, roi)
            batch  = tensor.unsqueeze(0).to(device)     # (1, 3, 384, 384)
            if use_fp16:
                batch = batch.half()
            t0 = time.perf_counter()
            with torch.no_grad():
                feat = spatial_model(batch).squeeze(0)  # (1152,)
            spatial_times.append(time.perf_counter() - t0)
            feature_deque.append(feat)

            # ── Object detection on the same sampled frame ────────────────────
            if enable_fusion:
                t0 = time.perf_counter()
                kps       = get_person_keypoints(frame, yolo)
                kp_radius = OBJ_KP_RADIUS_FRAC * max(roi[2] - roi[0], roi[3] - roi[1])
                tok, box, conf = detect_object_token(frame, yolo_obj, roi, kps, kp_radius)
                object_times.append(time.perf_counter() - t0)
                object_deque.append((tok, conf))

                # display de-flicker: only show a box for a persistent detection
                disp = persistent_token(object_deque)
                if disp is not None and tok == disp:
                    current_obj, current_obj_box = disp, box
                else:
                    current_obj, current_obj_box = None, None

            # ── Run temporal model once deque is full ─────────────────────────
            if len(feature_deque) == WINDOW_FRAMES:
                seq = torch.stack(list(feature_deque)).unsqueeze(0)  # (1, 16, 1152)
                t0 = time.perf_counter()
                with torch.no_grad():
                    logits = temporal_model(seq)
                temporal_times.append(time.perf_counter() - t0)
                probs = torch.softmax(logits, dim=-1).float().cpu().squeeze(0)  # (3,)

                # ── Fuse object prior into the temporal softmax ───────────────
                if enable_fusion and len(object_deque) == WINDOW_FRAMES:
                    prior = compute_object_prior(object_deque)
                    probs = fuse_probs(probs, prior, fusion_weight)

                current_cls   = int(probs.argmax())
                current_probs = probs
                all_predictions.append(CLASSES[current_cls])

                if not initialized:
                    initialized = True
                    print(f"  Initialized at frame {fidx}  "
                          f"(first prediction: {CLASSES[current_cls]})", flush=True)

        # ── Write frame with appropriate overlay ──────────────────────────────
        if initialized:
            draw_overlay(frame, current_cls, current_probs, roi, fidx, fps,
                         obj_token=current_obj if enable_fusion else None,
                         obj_box=current_obj_box if enable_fusion else None)
        else:
            draw_init_overlay(frame, roi, fidx, fps, len(feature_deque))
        writer.write(frame)

        if (fidx + 1) % 300 == 0:
            print(f"  {fidx + 1}/{total_frames} frames …", flush=True)

    total_wall = time.perf_counter() - t_start
    writer.release()
    cap.release()

    # ── Majority vote + summary ───────────────────────────────────────────────
    if not all_predictions:
        print("No predictions made (video too short for one full window).")
        return

    vote  = Counter(all_predictions)
    final = vote.most_common(1)[0][0]
    total = len(all_predictions)

    video_duration  = total_frames / fps
    proc_fps        = total_frames / total_wall
    realtime_factor = video_duration / total_wall
    avg_spatial_ms  = np.mean(spatial_times)  * 1000
    avg_temporal_ms = np.mean(temporal_times) * 1000
    budget_ms       = (STEP / fps) * 1000      # time budget per inference at real-time

    print("\n" + "═" * 42)
    print(f"  FINAL PREDICTION : {final}  ({vote[final]}/{total} windows)")
    print("─" * 42)
    for cls in CLASSES:
        n = vote.get(cls, 0)
        print(f"  {cls:<10}  {n:>7}  {n / total * 100:>5.1f}%")
    print("─" * 42)
    print(f"  Video duration   : {video_duration:.1f}s  ({total_frames} frames @ {fps:.1f} fps)")
    print(f"  Processing time  : {total_wall:.1f}s")
    print(f"  Processing speed : {proc_fps:.1f} fps  (real-time = {fps:.1f} fps)")
    print(f"  Real-time factor : {realtime_factor:.2f}x  "
          f"({'faster' if realtime_factor >= 1.0 else 'SLOWER'} than real-time)")
    print("─" * 42)
    print(f"  Avg spatial  inference : {avg_spatial_ms:.1f} ms / frame sampled")
    print(f"  Avg temporal inference : {avg_temporal_ms:.2f} ms / window")
    if object_times:
        avg_object_ms = np.mean(object_times) * 1000
        print(f"  Avg object   detection : {avg_object_ms:.1f} ms / frame sampled  (fusion W={fusion_weight})")
    print(f"  Time budget (real-time): {budget_ms:.1f} ms / {STEP} frames")
    if avg_spatial_ms <= budget_ms:
        print(f"  → Spatial fits real-time budget  ✓")
    else:
        print(f"  → Spatial EXCEEDS real-time budget by {avg_spatial_ms - budget_ms:.1f} ms")
    print("═" * 42)
    print(f"\nDone.  Output: {output_video}")

# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DriveGuard — single-camera driver activity inference (Safe / Drink / Phone)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-download weights + sample video and run:
  python infer.py

  # Custom paths:
  python infer.py \\
      --video            ./clip.mp4 \\
      --spatial_weights  ./vit_spatial_model_v1.pth \\
      --temporal_weights ./temporal_head_model.pth \\
      --output_video     ./out.mp4
        """,
    )
    parser.add_argument('--video',            default=None,
                        help='Input video path (default: auto-download sample)')
    parser.add_argument('--spatial_weights',  default=None,
                        help='Spatial model weights (default: auto-download)')
    parser.add_argument('--temporal_weights', default=None,
                        help='Temporal model weights (default: auto-download)')
    parser.add_argument('--output_video', default='inference_output.mp4',
                        help='Output video path (default: inference_output.mp4)')
    parser.add_argument('--fusion_weight', type=float, default=FUSION_WEIGHT,
                        help=f'Object-prior fusion weight W (default: {FUSION_WEIGHT}; 0 = temporal only)')
    parser.add_argument('--no_object_detection', action='store_true',
                        help='Disable object-detection fusion (temporal-only baseline)')
    parser.add_argument('--obj_weights', default=OBJ_WEIGHTS,
                        help=f'COCO object detector weights (default: {OBJ_WEIGHTS})')
    args = parser.parse_args()

    spatial_weights  = args.spatial_weights  or _ensure_downloaded('vit_spatial_model_v1.pth')
    temporal_weights = args.temporal_weights or _ensure_downloaded('temporal_head_model.pth')
    video_path       = args.video            or _ensure_downloaded('sample_video.mp4')

    run_inference(
        video_path       = video_path,
        spatial_weights  = spatial_weights,
        temporal_weights = temporal_weights,
        output_video     = args.output_video,
        enable_fusion    = not args.no_object_detection,
        fusion_weight    = args.fusion_weight,
        obj_weights      = args.obj_weights,
    )


if __name__ == '__main__':
    main()
