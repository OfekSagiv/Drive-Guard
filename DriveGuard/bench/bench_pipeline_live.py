#!/usr/bin/env python3
"""
Faithful live-pacing benchmark: reproduces the actual run_live() consumer loop
(unconditional fidx increment, no sleep, non-blocking FreshestFrame.read())
against a producer thread that paces frame delivery at the source's real fps
-- i.e. simulates a genuine 30fps camera feed instead of a file read as fast
as the decoder allows. Uses the real FreshestFrame class from
DriveGuard/infer_live_ir.py unmodified; only the capture source is paced.

Measures, for fusion ON and OFF:
  - real wall-clock seconds between consecutive STEP-processed samples
  - how many real camera frames were produced (and dropped) between samples
  - the resulting real-world span of a 16-sample temporal window

Known limitation of this method (see docs/project_book.md Section 5.2): the
paced producer thread itself was observed to sustain only ~6.5 FPS instead
of the intended 30 FPS on this machine, likely due to GIL contention with
the heavy synchronous torch/OpenCV calls in the consumer thread. A real RTSP
decoder may or may not be similarly affected -- treat these numbers as a
measured lower bound on the real-time gap, not an upper bound.
"""
import sys, time, threading, statistics as stats
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]   # bench/ → DriveGuard/ → repo root
sys.path.insert(0, str(REPO / "DriveGuard"))

import cv2
import torch
import infer_live_ir as m

VIDEO = str(REPO / "inference_output.mp4")
REAL_SECONDS_PER_PASS = 45  # wall-clock seconds to run each pass


class PacedCapture:
    """Wraps cv2.VideoCapture so .read() only returns a new frame at the
    source's real fps, looping the file to simulate a continuous live feed.
    Tracks produced_count so the consumer can measure drop rate."""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.interval = 1.0 / fps
        self.fps = fps
        self.next_t = time.perf_counter()
        self.produced_count = 0
        self.lock = threading.Lock()

    def read(self):
        now = time.perf_counter()
        wait = self.next_t - now
        if wait > 0:
            time.sleep(wait)
        self.next_t = max(self.next_t + self.interval, time.perf_counter())
        ok, frame = self.cap.read()
        if not ok:  # loop the file
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        with self.lock:
            self.produced_count += 1
        return ok, frame

    def release(self):
        self.cap.release()


def run_pass(models, enable_fusion: bool):
    device, use_fp16 = models['device'], models['use_fp16']
    yolo, yolo_obj = models['yolo'], models['yolo_obj']
    spatial_model, temporal_model = models['spatial'], models['temporal']

    paced = PacedCapture(VIDEO)
    reader = m.FreshestFrame(paced)   # real production class, unmodified

    roi = None
    img_h = img_w = None
    feature_deque = m.deque(maxlen=m.WINDOW_FRAMES)
    object_deque  = m.deque(maxlen=m.WINDOW_FRAMES)

    sample_times = []       # wall-clock perf_counter() at each STEP-processed sample
    sample_produced = []    # paced.produced_count at each STEP-processed sample
    fidx = 0
    t_end = time.perf_counter() + REAL_SECONDS_PER_PASS

    # wait for first frame
    while True:
        ok, raw = reader.read()
        if ok:
            break
        time.sleep(0.01)

    while time.perf_counter() < t_end:
        ok, raw = reader.read()
        if not ok:
            continue
        frame = m.downscale(raw, m.PROC_WIDTH)
        if img_h is None:
            img_h, img_w = frame.shape[:2]

        if fidx % m.CYCLE_SIZE == 0:
            new_roi = m.detect_roi_from_frame(frame, yolo, img_h, img_w)
            if new_roi is not None:
                roi = new_roi
            elif roi is None:
                roi = (0, 0, img_w, img_h)

        if fidx % m.STEP == 0 and roi is not None:
            batch = m.crop_and_preprocess(frame, roi).unsqueeze(0).to(device)
            if use_fp16:
                batch = batch.half()
            with torch.no_grad():
                feat = spatial_model(batch).squeeze(0)
            if device.type == "mps":
                torch.mps.synchronize()
            feature_deque.append(feat)

            if enable_fusion:
                kps = m.get_person_keypoints(frame, yolo)
                kp_radius = m.OBJ_KP_RADIUS_FRAC * max(roi[2]-roi[0], roi[3]-roi[1])
                tok, box, conf = m.detect_object_token(frame, yolo_obj, roi, kps, kp_radius)
                object_deque.append((tok, conf))

            if len(feature_deque) == m.WINDOW_FRAMES:
                seq = torch.stack(list(feature_deque)).unsqueeze(0)
                with torch.no_grad():
                    logits = temporal_model(seq)
                temporal_probs = torch.softmax(logits, dim=-1).float().cpu().squeeze(0)
                if enable_fusion and len(object_deque) == m.WINDOW_FRAMES:
                    prior = m.compute_object_prior(object_deque)
                    _ = m.fuse_probs(temporal_probs, prior, m.FUSION_WEIGHT)

            sample_times.append(time.perf_counter())
            with paced.lock:
                sample_produced.append(paced.produced_count)

        fidx += 1

    reader.stop()

    # Derive: inter-sample real-world spacing, and real camera frames elapsed per sample
    gaps = [t2 - t1 for t1, t2 in zip(sample_times[:-1], sample_times[1:])]
    frame_gaps = [p2 - p1 for p1, p2 in zip(sample_produced[:-1], sample_produced[1:])]
    return dict(
        n_samples=len(sample_times),
        real_fps=paced.fps,
        mean_gap_ms=stats.mean(gaps)*1000 if gaps else float('nan'),
        median_gap_ms=stats.median(gaps)*1000 if gaps else float('nan'),
        mean_frames_per_sample=stats.mean(frame_gaps) if frame_gaps else float('nan'),
        total_fidx_iterations=fidx,
        total_real_frames_produced=paced.produced_count,
    )


def main():
    print("Loading models...")
    models = m.build_models(
        str(REPO / "pipelines/vit_transformer/vit_spatial_model_v1.pth"),
        str(REPO / "pipelines/vit_transformer/temporal_head_model.pth"),
    )
    print(f"  device={models['device']}  fp16={models['use_fp16']}\n")

    print(f"=== Pass 1: fusion OFF  ({REAL_SECONDS_PER_PASS}s real wall-clock) ===")
    off = run_pass(models, enable_fusion=False)
    for k, v in off.items():
        print(f"  {k}: {v}")

    print(f"\n=== Pass 2: fusion ON  ({REAL_SECONDS_PER_PASS}s real wall-clock) ===")
    on = run_pass(models, enable_fusion=True)
    for k, v in on.items():
        print(f"  {k}: {v}")

    nominal_step_ms = 1000 * m.STEP / off['real_fps']
    print(f"\nNominal STEP={m.STEP} spacing at {off['real_fps']:.0f} FPS = {nominal_step_ms:.1f} ms")
    print(f"Intended 16-frame window span = {nominal_step_ms*m.WINDOW_FRAMES/1000:.2f} s")

    for name, r in [("fusion OFF", off), ("fusion ON", on)]:
        real_span = r['mean_gap_ms'] * m.WINDOW_FRAMES / 1000
        print(f"\n[{name}] actual mean inter-sample gap = {r['mean_gap_ms']:.1f} ms "
              f"({r['mean_gap_ms']/nominal_step_ms:.1f}x nominal)")
        print(f"[{name}] real camera frames elapsed per processed sample = {r['mean_frames_per_sample']:.1f} "
              f"(vs. nominal STEP={m.STEP})")
        print(f"[{name}] real-world span of a 16-sample window = {real_span:.2f} s "
              f"(vs. intended {nominal_step_ms*m.WINDOW_FRAMES/1000:.2f} s)")


if __name__ == "__main__":
    main()
