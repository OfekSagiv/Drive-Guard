#!/usr/bin/env python3
"""
DriveGuard CNN+LSTM Inference Script
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

Usage:
  python infer.py

  python infer.py \\
      --video            /path/to/video.mp4 \\
      --spatial_weights  /path/to/efficientnet_b4_spatial_model_v1.pth \\
      --temporal_weights /path/to/lstm_temporal_head_model.pth \\
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
# Google Drive assets  (set file IDs after uploading weights to Drive)
# ──────────────────────────────────────────────────────────────────────────────

_DRIVE_ASSETS = {
    'efficientnet_b4_spatial_model_v1.pth': '1rPQR6I4nCAI1A6xcNMCJbrN1nWx3AKJG', 
    'lstm_temporal_head_model.pth'         : '14jpXUMKe1KoAp7Tswy8Oc8ZBy7gnW9Rv',
    'sample_video.mp4'                     : '1JhhimPGppUqlsa-Yqi9IsPWl66-nGsjl',
}


def _ensure_downloaded(filename: str) -> str:
    """Download file from Google Drive if not already present. Returns local path."""
    path = Path(__file__).parent / filename
    if path.exists():
        return str(path)
    file_id = _DRIVE_ASSETS.get(filename)
    if not file_id:
        print(f"ERROR: '{filename}' not found locally and no Drive file ID is set.")
        print(f"       Place the file next to this script or set its ID in _DRIVE_ASSETS.")
        sys.exit(1)
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

CLASSES      = ['Drink', 'Phone', 'Safe']
CLASS_COLORS = {                          # BGR for OpenCV
    0: (0,   165, 255),  # Drink — orange
    1: (0,   0,   255),  # Phone — red
    2: (0,   200, 0  ),  # Safe  — green
}

WINDOW_FRAMES = 16    # temporal model input length
STEP          = 6     # sample one frame every STEP frames
CYCLE_SIZE    = 90    # ROI refresh interval (frames)
IMG_SIZE      = 380   # EfficientNet-B4 native resolution
ROI_PADDING   = 0.08
YOLO_CONF     = 0.25

# ImageNet normalisation — must match EfficientNet-B4 training
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────────────────────────────────────
# Temporal Model  (must match lstm_temporal_head.ipynb exactly)
# ──────────────────────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """
    Input projection + 2-layer bidirectional LSTM.
    Input:  (B, T=16, D=1792)
    Output: (B, hidden_dim*2=1024)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        H = cfg['hidden_dim']
        self.proj = nn.Sequential(
            nn.Linear(cfg['input_dim'], H),
            nn.LayerNorm(H),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size    = H,
            hidden_size   = H,
            num_layers    = cfg['num_layers'],
            batch_first   = True,
            bidirectional = cfg['bidir'],
            dropout       = cfg['dropout'] if cfg['num_layers'] > 1 else 0.0,
        )
        self.bidir = cfg['bidir']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                       # (B, T, H)
        _, (h, _) = self.lstm(x)               # h: (n_layers * dirs, B, H)
        if self.bidir:
            return torch.cat([h[-2], h[-1]], dim=1)  # (B, H*2)
        return h[-1]                                  # (B, H)


class DriveGuardLSTM(nn.Module):
    """
    LSTMEncoder + Dropout + Linear classifier.
    Input:  (B, T=16, D=1792)
    Output: (B, num_classes) logits
    Must match lstm_temporal_head.ipynb exactly.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        out_dim      = cfg['hidden_dim'] * (2 if cfg['bidir'] else 1)
        self.encoder = LSTMEncoder(cfg)
        self.drop    = nn.Dropout(cfg['dropout'])
        self.head    = nn.Linear(out_dim, cfg['num_classes'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.encoder(x)))


TEMPORAL_CFG = {
    'input_dim'  : 1792,
    'hidden_dim' : 512,
    'num_layers' : 2,
    'bidir'      : True,
    'dropout'    : 0.3,
    'noise_std'  : 0.0,   # disabled at inference
    'num_classes': 3,
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
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

_PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def crop_and_preprocess(frame_bgr: np.ndarray, roi: tuple) -> torch.Tensor:
    """Crop ROI from a BGR frame and return a preprocessed (3, 380, 380) tensor."""
    x1, y1, x2, y2 = roi
    crop = frame_bgr[y1:y2, x1:x2]
    img  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return _PREPROCESS(img)

# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_spatial_model(weights_path: str, device: torch.device, use_fp16: bool) -> nn.Module:
    """Load EfficientNet-B4, strip classifier head → 1792-dim feature extractor."""
    print(f"  Spatial  model ← {weights_path}")
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=3)
    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state)
    model.reset_classifier(0)
    model = model.to(device)
    if use_fp16:
        model = model.half()
    return model.eval()


def load_temporal_model(weights_path: str, device: torch.device, use_fp16: bool,
                        cfg: dict = None) -> nn.Module:
    """Load DriveGuardLSTM from checkpoint."""
    print(f"  Temporal model ← {weights_path}")
    model = DriveGuardLSTM(cfg or TEMPORAL_CFG)
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
) -> np.ndarray:
    """Annotate a frame with class prediction, confidence bars, and ROI box."""
    _, w = frame.shape[:2]
    color    = CLASS_COLORS[cls_idx]
    cls_name = CLASSES[cls_idx]
    conf     = probs[cls_idx].item()

    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    bar_x      = w - 170
    bar_w      = 140
    bar_h      = 22
    pad_y      = 12
    banner_w, banner_h = 230, 75

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (banner_w, banner_h), (15, 15, 15), -1)
    cv2.rectangle(overlay, (bar_x - 8, pad_y - 4),
                  (w - 4, pad_y + len(CLASSES) * (bar_h + 8) + 4), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, cls_name,
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
    cv2.putText(frame, f'{conf * 100:.1f}%  t={frame_idx / fps:.1f}s',
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)

    for i, (cname, p) in enumerate(zip(CLASSES, probs.tolist())):
        by   = pad_y + i * (bar_h + 8)
        fill = int(bar_w * p)
        c    = CLASS_COLORS[i]
        cv2.rectangle(frame, (bar_x, by),       (bar_x + bar_w, by + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, by),       (bar_x + fill,  by + bar_h), c, -1)
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
    initialized   = False
    all_predictions: list = []

    current_cls   = CLASSES.index('Safe')
    current_probs = torch.zeros(len(CLASSES))
    current_probs[current_cls] = 1.0

    spatial_times:  list = []
    temporal_times: list = []

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
            batch  = tensor.unsqueeze(0).to(device)   # (1, 3, 380, 380)
            if use_fp16:
                batch = batch.half()
            t0 = time.perf_counter()
            with torch.no_grad():
                feat = spatial_model(batch).squeeze(0)  # (1792,)
            spatial_times.append(time.perf_counter() - t0)
            feature_deque.append(feat)

            # ── Run temporal model once deque is full ─────────────────────────
            if len(feature_deque) == WINDOW_FRAMES:
                seq = torch.stack(list(feature_deque)).unsqueeze(0)  # (1, 16, 1792)
                t0 = time.perf_counter()
                with torch.no_grad():
                    logits = temporal_model(seq)
                temporal_times.append(time.perf_counter() - t0)
                probs = torch.softmax(logits, dim=-1).float().cpu().squeeze(0)  # (3,)

                current_cls   = int(probs.argmax())
                current_probs = probs
                all_predictions.append(CLASSES[current_cls])

                if not initialized:
                    initialized = True
                    print(f"  Initialized at frame {fidx}  "
                          f"(first prediction: {CLASSES[current_cls]})", flush=True)

        # ── Write frame with appropriate overlay ──────────────────────────────
        if initialized:
            draw_overlay(frame, current_cls, current_probs, roi, fidx, fps)
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
    budget_ms       = (STEP / fps) * 1000

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
        description='DriveGuard CNN+LSTM — single-camera driver activity inference (Safe / Drink / Phone)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer.py

  python infer.py \\
      --video            ./clip.mp4 \\
      --spatial_weights  ./efficientnet_b4_spatial_model_v1.pth \\
      --temporal_weights ./lstm_temporal_head_model.pth \\
      --output_video     ./out.mp4
        """,
    )
    parser.add_argument('--video',            default=None,
                        help='Input video path (default: auto-download sample)')
    parser.add_argument('--spatial_weights',  default=None,
                        help='EfficientNet-B4 spatial weights (default: auto-download)')
    parser.add_argument('--temporal_weights', default=None,
                        help='LSTM temporal weights (default: auto-download)')
    parser.add_argument('--output_video', default='inference_output.mp4',
                        help='Output video path (default: inference_output.mp4)')
    args = parser.parse_args()

    spatial_weights  = args.spatial_weights  or _ensure_downloaded('efficientnet_b4_spatial_model_v1.pth')
    temporal_weights = args.temporal_weights or _ensure_downloaded('lstm_temporal_head_model.pth')
    video_path       = args.video            or _ensure_downloaded('sample_video.mp4')

    run_inference(
        video_path       = video_path,
        spatial_weights  = spatial_weights,
        temporal_weights = temporal_weights,
        output_video     = args.output_video,
    )


if __name__ == '__main__':
    main()
