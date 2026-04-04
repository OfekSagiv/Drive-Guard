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

Usage:
  python infer.py \\
      --video            /path/to/a_column_co_driver.mp4 \\
      --spatial_weights  /path/to/best_model_fused.pth \\
      --temporal_weights /path/to/best_stage4_single_cam_model.pth

  # Custom output path:
  python infer.py \\
      --video ./clip.mp4 --spatial_weights ./spatial.pth --temporal_weights ./temporal.pth \\
      --output_video ./out.mp4
"""

import argparse
import sys
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
    'hidden_dim'     : 512,
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
    initialized   = False                        # True once deque has WINDOW_FRAMES features
    all_predictions: list = []

    current_cls   = CLASSES.index('Safe')
    current_probs = torch.zeros(len(CLASSES))
    current_probs[current_cls] = 1.0

    # ── Streaming pass ────────────────────────────────────────────────────────
    print(f"Processing …  (writing to {output_video})", flush=True)

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
            with torch.no_grad():
                feat = spatial_model(batch).squeeze(0)  # (1152,)
            feature_deque.append(feat)

            # ── Run temporal model once deque is full ─────────────────────────
            if len(feature_deque) == WINDOW_FRAMES:
                seq = torch.stack(list(feature_deque)).unsqueeze(0)  # (1, 16, 1152)
                with torch.no_grad():
                    logits = temporal_model(seq)
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

    writer.release()
    cap.release()

    # ── Majority vote + summary ───────────────────────────────────────────────
    if not all_predictions:
        print("No predictions made (video too short for one full window).")
        return

    vote  = Counter(all_predictions)
    final = vote.most_common(1)[0][0]
    total = len(all_predictions)

    print("\n" + "═" * 32)
    print(f"  FINAL PREDICTION : {final}  ({vote[final]}/{total} windows)")
    print("─" * 32)
    for cls in CLASSES:
        n = vote.get(cls, 0)
        print(f"  {cls:<10}  {n:>7}  {n / total * 100:>5.1f}%")
    print("═" * 32)
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
  python infer.py \\
      --video            ./a_column_co_driver.mp4 \\
      --spatial_weights  ./best_model_fused.pth \\
      --temporal_weights ./best_stage4_single_cam_model.pth

  python infer.py \\
      --video ./clip.mp4 --spatial_weights ./spatial.pth --temporal_weights ./temporal.pth \\
      --output_video ./out.mp4
        """,
    )
    parser.add_argument('--video',            required=True)
    parser.add_argument('--spatial_weights',  required=True)
    parser.add_argument('--temporal_weights', required=True)
    parser.add_argument('--output_video', default='inference_output.mp4',
                        help='Output video path (default: inference_output.mp4)')
    args = parser.parse_args()

    run_inference(
        video_path       = args.video,
        spatial_weights  = args.spatial_weights,
        temporal_weights = args.temporal_weights,
        output_video     = args.output_video,
    )


if __name__ == '__main__':
    main()
