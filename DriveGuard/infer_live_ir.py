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
from ultralytics import YOLO, YOLOE

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
# Object-detection fusion  (YOLOE-26-L soft prior)
# ──────────────────────────────────────────────────────────────────────────────
OBJ_CONF           = 0.25   # confidence threshold (tuned for IR, 0.6–0.85 signals observed)
OBJ_SMOOTHING      = 0.5    # Laplace smoothing — neutral floor for all classes
FUSION_WEIGHT      = 0.6    # W in log-linear fusion; starting point, tune via W sweep
OBJ_KP_RADIUS_FRAC = 0.33   # object center must be within frac×ROI-size of a keypoint
KP_CONF            = 0.30   # minimum keypoint confidence to use
RELEVANT_KP        = {0, 1, 2, 3, 4, 7, 8, 9, 10}  # nose, eyes, ears, elbows, wrists
OBJ_PERSIST_K      = 3      # de-flicker: look at last K detection samples
OBJ_PERSIST_MIN    = 2      # de-flicker: draw box only if token seen in ≥ MIN of last K

# YOLOE-26-L open-vocabulary prompts → DriveGuard tokens
YW_PHONE_CLASSES = ['smartphone', 'cell phone']
YW_DRINK_CLASSES = ['mug', 'cup', 'bottle', 'drinking glass']
YW_ALL_CLASSES   = YW_PHONE_CLASSES + YW_DRINK_CLASSES   # 6 prompts
YW_PHONE_SET     = set(YW_PHONE_CLASSES)
YW_DRINK_SET     = set(YW_DRINK_CLASSES)

DRINK_IDX = CLASSES.index('Drink')
PHONE_IDX = CLASSES.index('Phone')
SAFE_IDX  = CLASSES.index('Safe')

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
# Object-detection fusion helpers
# ──────────────────────────────────────────────────────────────────────────────

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
    """Drop boxes mostly swallowed by a higher-confidence box.

    Plain IoU-NMS misses small boxes nested inside large ones because the large
    box's extra area dominates the union. This checks intersection/own_area instead.
    Input dets must be sorted by confidence descending.
    """
    kept = []
    for d in dets:
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


def get_person_keypoints(frame: np.ndarray, pose_model) -> np.ndarray:
    """Return (N, 2) xy array of hand/face keypoints for the most confident person."""
    results = pose_model(frame, conf=YOLO_CONF, verbose=False)
    if (not results or results[0].keypoints is None
            or len(results[0].boxes) == 0):
        return np.empty((0, 2))
    best = int(np.argmax(results[0].boxes.conf.cpu().numpy()))
    kp   = results[0].keypoints.data.cpu().numpy()[best]   # (17, 3)
    pts  = [(x, y) for i, (x, y, c) in enumerate(kp)
            if i in RELEVANT_KP and c >= KP_CONF]
    return np.array(pts, dtype=np.float32) if pts else np.empty((0, 2))


def detect_object_token(frame: np.ndarray, yolo_obj, roi: tuple,
                        keypoints: np.ndarray = None, kp_radius: float = None):
    """Detect phone/drink near driver's hands/face using YOLOE-26-L.

    Returns (token, box, conf): token ∈ {'phone', 'drink', None}.
    Gates: (1) box center inside ROI; (2) center within kp_radius of a keypoint.
    agnostic_nms=True is set in the predict call to suppress cross-prompt duplicates.
    suppress_contained() is applied after parsing to catch nested boxes IoU-NMS misses.
    """
    rx1, ry1, rx2, ry2 = roi
    results = yolo_obj.predict(frame, conf=OBJ_CONF, iou=0.5,
                               agnostic_nms=True, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None, None, 0.0

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    clses = results[0].boxes.cls.cpu().numpy().astype(int)
    names = results[0].names   # {int: str} — YOLOE class names from set_classes()

    # Parse into list of dicts, sort by conf descending, apply containment filter
    dets = sorted(
        [{'name': names[c], 'conf': float(cf), 'xyxy': (int(x1), int(y1), int(x2), int(y2))}
         for (x1, y1, x2, y2), cf, c in zip(boxes, confs, clses)],
        key=lambda d: d['conf'], reverse=True,
    )
    dets = suppress_contained(dets, 0.8)

    use_kp = keypoints is not None and len(keypoints) > 0 and kp_radius is not None
    best_token, best_box, best_conf = None, None, -1.0

    for d in dets:
        bx1, by1, bx2, by2 = d['xyxy']
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):   # gate 1: ROI
            continue
        if use_kp:                                          # gate 2: keypoint proximity
            dist = np.min(np.hypot(keypoints[:, 0] - cx, keypoints[:, 1] - cy))
            if dist > kp_radius:
                continue
        if d['conf'] <= best_conf:
            continue
        best_conf  = d['conf']
        best_token = 'phone' if d['name'] in YW_PHONE_SET else 'drink'
        best_box   = (bx1, by1, bx2, by2)

    return best_token, best_box, (best_conf if best_token else 0.0)


def compute_object_prior(object_deque, window_size: int = WINDOW_FRAMES) -> torch.Tensor:
    """Build a normalized object prior from the 16-frame detection window.

    Formula: w[class] += Σconf_for_class / window_size
    This equals mean_conf × detection_rate, naturally bounded to [0, 1].
    Consecutive detections are correlated, so linear accumulation would inflate
    the prior — dividing by window_size corrects for this.
    None frames add nothing; Laplace smoothing supplies the neutral floor.
    All-None window → [0.33, 0.33, 0.33] → log-linear fusion = temporal-only.
    """
    w = torch.full((len(CLASSES),), OBJ_SMOOTHING)
    for tok, conf in object_deque:
        if tok == 'phone':
            w[PHONE_IDX] += conf / window_size
        elif tok == 'drink':
            w[DRINK_IDX] += conf / window_size
    return w / w.sum()


def persistent_token(object_deque) -> str | None:
    """De-flicker: return latest token only if it appears ≥ OBJ_PERSIST_MIN of last K."""
    recent = [tok for tok, _ in list(object_deque)[-OBJ_PERSIST_K:]]
    if not recent or recent[-1] is None:
        return None
    return recent[-1] if recent.count(recent[-1]) >= OBJ_PERSIST_MIN else None


def fuse_probs(temporal_probs: torch.Tensor, prior: torch.Tensor,
               weight: float, eps: float = 1e-6) -> torch.Tensor:
    """Log-linear (product-of-experts) fusion: softmax(log p_temporal + W·log prior).

    W=0 or uniform prior → reproduces temporal distribution exactly.
    """
    log_p = torch.log(temporal_probs + eps) + weight * torch.log(prior + eps)
    return torch.softmax(log_p, dim=-1)

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
    Call from a background thread — takes ~30s on first run (includes YOLOE-26-L download).
    Returns: {'device', 'use_fp16', 'yolo', 'yolo_obj', 'spatial', 'temporal'}
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
    print("  Object detector ← yoloe-26l-seg.pt  (YOLOE-26-L, auto-download if absent)")
    yolo_obj = YOLOE('yoloe-26l-seg.pt')
    yolo_obj.set_classes(YW_ALL_CLASSES)
    spatial_model  = load_spatial_model(spatial_weights, device, use_fp16)
    temporal_model = load_temporal_model(temporal_weights, device, use_fp16)
    print("  Models ready.\n")

    return dict(device=device, use_fp16=use_fp16,
                yolo=yolo, yolo_obj=yolo_obj,
                spatial=spatial_model, temporal=temporal_model)

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
    on_prediction=None,        # callback(cls_idx, temporal_probs, prior, fused_probs, initialized)
    should_run=None,           # callable() → bool; loop exits when False
    on_status=None,            # callback(text, color)
    # ── Fusion params ─────────────────────────────────────────────────────────
    enable_fusion: bool = True,
    fusion_weight: float = FUSION_WEIGHT,
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
        yolo           = models['yolo']
        yolo_obj       = models.get('yolo_obj')
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
        yolo     = YOLO(str(_WEIGHTS_DIR / 'yolov8n-pose.pt'))
        yolo_obj = None
        if enable_fusion:
            yolo_obj = YOLOE('yoloe-26l-seg.pt')
            yolo_obj.set_classes(YW_ALL_CLASSES)
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
    object_deque    = deque(maxlen=WINDOW_FRAMES)
    current_obj     = None
    current_obj_box = None
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

                # ── Object detection (synchronized with feature sampling) ──────
                if enable_fusion and yolo_obj is not None:
                    kps       = get_person_keypoints(frame, yolo)
                    kp_radius = OBJ_KP_RADIUS_FRAC * max(roi[2] - roi[0],
                                                          roi[3] - roi[1])
                    tok, box, conf = detect_object_token(
                        frame, yolo_obj, roi, kps, kp_radius)
                    object_deque.append((tok, conf))
                    disp = persistent_token(object_deque)
                    if disp is not None and tok == disp:
                        current_obj, current_obj_box = disp, box
                    else:
                        current_obj, current_obj_box = None, None

                # Report init progress to the UI on each new feature
                if not initialized:
                    n = len(feature_deque)
                    _status(f'Initializing…  {n} / {WINDOW_FRAMES} features', '#ffaa00')

                if len(feature_deque) == WINDOW_FRAMES:
                    seq = torch.stack(list(feature_deque)).unsqueeze(0)
                    with torch.no_grad():
                        logits = temporal_model(seq)
                    temporal_probs = torch.softmax(logits, dim=-1).float().cpu().squeeze(0)

                    # Fusion
                    prior       = None
                    fused_probs = None
                    if enable_fusion and yolo_obj is not None and len(object_deque) == WINDOW_FRAMES:
                        prior       = compute_object_prior(object_deque)
                        fused_probs = fuse_probs(temporal_probs, prior, fusion_weight)

                    probs         = fused_probs if fused_probs is not None else temporal_probs
                    current_cls   = int(probs.argmax())
                    current_probs = probs
                    all_predictions.append(CLASSES[current_cls])
                    if not initialized:
                        initialized = True
                        print(f"  Initialized  (first: {CLASSES[current_cls]})", flush=True)
                    if on_prediction is not None:
                        on_prediction(current_cls, temporal_probs, prior, fused_probs, initialized)

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
