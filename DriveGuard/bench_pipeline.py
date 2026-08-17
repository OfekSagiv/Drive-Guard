#!/usr/bin/env python3
"""
Benchmark the DriveGuard live-inference pipeline stage-by-stage:
YOLO pose ROI, SigLIP spatial encoder, YOLOE-26-L object detector +
keypoint gating, temporal Transformer, and fusion, on real local hardware,
using the exact functions from DriveGuard/infer_live_ir.py (not a
reimplementation). Frames are read from a local video file instead of the
live RTSP stream, since compute cost is independent of the frame source;
RTSP/network overhead is a separate, additive cost noted in the report.

Note: this script measures raw per-stage compute time by reading frames
sequentially from a file (no pacing, no frame drops). It does NOT reflect
how the live app behaves under FreshestFrame's non-blocking, no-queue
frame delivery -- see bench_pipeline_live.py for that.
"""
import sys, time, statistics as stats
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]   # bench/ → DriveGuard/ → repo root
sys.path.insert(0, str(REPO / "DriveGuard"))

import cv2
import torch

import infer_live_ir as m

VIDEO = str(REPO / "inference_output.mp4")
N_FRAMES = 900  # ~30s at 30fps source -> covers 10 ROI cycles, 150 STEP samples, ~8 full 16-window fills

def main():
    print("Loading models (spatial, temporal, YOLO pose, YOLOE-26-L + MobileCLIP)...")
    t0 = time.perf_counter()
    models = m.build_models(
        str(REPO / "pipelines/vit_transformer/vit_spatial_model_v1.pth"),
        str(REPO / "pipelines/vit_transformer/temporal_head_model.pth"),
    )
    print(f"  build_models() took {time.perf_counter()-t0:.1f}s")
    device, use_fp16 = models['device'], models['use_fp16']
    yolo, yolo_obj = models['yolo'], models['yolo_obj']
    spatial_model, temporal_model = models['spatial'], models['temporal']
    print(f"  device={device}  fp16={use_fp16}")

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("Could not open video"); sys.exit(1)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"  source video fps={src_fps:.1f}")

    roi = None
    img_h = img_w = None
    feature_deque = m.deque(maxlen=m.WINDOW_FRAMES)
    object_deque  = m.deque(maxlen=m.WINDOW_FRAMES)

    t_roi, t_spatial, t_kp, t_obj, t_temporal, t_fusion = [], [], [], [], [], []
    t_step_fusion_off, t_step_fusion_on = [], []  # total per-STEP-frame cost, with/without OD

    fidx = 0
    while fidx < N_FRAMES:
        ok, raw = cap.read()
        if not ok:
            break
        frame = m.downscale(raw, m.PROC_WIDTH)
        if img_h is None:
            img_h, img_w = frame.shape[:2]

        if fidx % m.CYCLE_SIZE == 0:
            t0 = time.perf_counter()
            new_roi = m.detect_roi_from_frame(frame, yolo, img_h, img_w)
            t_roi.append(time.perf_counter() - t0)
            if new_roi is not None:
                roi = new_roi
            elif roi is None:
                roi = (0, 0, img_w, img_h)

        if fidx % m.STEP == 0 and roi is not None:
            step_t0 = time.perf_counter()

            t0 = time.perf_counter()
            batch = m.crop_and_preprocess(frame, roi).unsqueeze(0).to(device)
            if use_fp16:
                batch = batch.half()
            with torch.no_grad():
                feat = spatial_model(batch).squeeze(0)
            if device.type == "mps":
                torch.mps.synchronize()
            dt_spatial = time.perf_counter() - t0
            t_spatial.append(dt_spatial)
            feature_deque.append(feat)

            t0 = time.perf_counter()
            kps = m.get_person_keypoints(frame, yolo)
            dt_kp = time.perf_counter() - t0
            t_kp.append(dt_kp)

            kp_radius = m.OBJ_KP_RADIUS_FRAC * max(roi[2]-roi[0], roi[3]-roi[1])
            t0 = time.perf_counter()
            tok, box, conf = m.detect_object_token(frame, yolo_obj, roi, kps, kp_radius)
            dt_obj = time.perf_counter() - t0
            t_obj.append(dt_obj)
            object_deque.append((tok, conf))

            dt_temporal = 0.0
            if len(feature_deque) == m.WINDOW_FRAMES:
                seq = torch.stack(list(feature_deque)).unsqueeze(0)
                t0 = time.perf_counter()
                with torch.no_grad():
                    logits = temporal_model(seq)
                temporal_probs = torch.softmax(logits, dim=-1).float().cpu().squeeze(0)
                if device.type == "mps":
                    torch.mps.synchronize()
                dt_temporal = time.perf_counter() - t0
                t_temporal.append(dt_temporal)

                t0 = time.perf_counter()
                prior = m.compute_object_prior(object_deque)
                fused = m.fuse_probs(temporal_probs, prior, m.FUSION_WEIGHT)
                dt_fusion = time.perf_counter() - t0
                t_fusion.append(dt_fusion)

            t_step_fusion_on.append(dt_spatial + dt_kp + dt_obj + dt_temporal)
            t_step_fusion_off.append(dt_spatial + dt_temporal)

        fidx += 1

    cap.release()

    def rpt(name, arr, budget_ms=None):
        if not arr:
            print(f"{name}: no samples"); return
        ms = [x*1000 for x in arr]
        mean, med = stats.mean(ms), stats.median(ms)
        p95 = sorted(ms)[int(0.95*len(ms))-1] if len(ms) > 1 else ms[0]
        extra = f"   budget={budget_ms}ms  {'OK' if mean<budget_ms else 'OVER'}" if budget_ms else ""
        print(f"{name:28s} n={len(ms):4d}  mean={mean:7.1f}ms  median={med:7.1f}ms  p95={p95:7.1f}ms{extra}")

    print("\n=== Per-stage latency (Apple Silicon MPS, this machine) ===")
    rpt("YOLO pose ROI (per cycle)", t_roi)
    rpt("SigLIP spatial encode", t_spatial)
    rpt("Pose keypoints (for OD gate)", t_kp)
    rpt("YOLOE-26-L object detect", t_obj)
    rpt("Temporal Transformer", t_temporal)
    rpt("Fusion (log-linear)", t_fusion)

    step_budget_ms = 1000 * m.STEP / src_fps
    print(f"\nReal-time budget per STEP={m.STEP}-frame cycle at {src_fps:.0f} FPS source = {step_budget_ms:.1f} ms")
    rpt("TOTAL per-STEP (fusion OFF)", t_step_fusion_off, step_budget_ms)
    rpt("TOTAL per-STEP (fusion ON)", t_step_fusion_on, step_budget_ms)

    off_mean = stats.mean(t_step_fusion_off) * 1000
    on_mean  = stats.mean(t_step_fusion_on) * 1000
    print(f"\nEffective real-time multiplier, fusion OFF: {step_budget_ms/off_mean:.2f}x")
    print(f"Effective real-time multiplier, fusion ON : {step_budget_ms/on_mean:.2f}x")
    print(f"Fusion overhead added per STEP-cycle: {on_mean-off_mean:.1f} ms ({(on_mean/off_mean-1)*100:.0f}% slower)")

if __name__ == "__main__":
    main()
