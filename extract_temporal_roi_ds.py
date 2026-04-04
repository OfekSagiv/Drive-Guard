import os
import cv2
import pandas as pd
import numpy as np
import torch
import shutil
from tqdm import tqdm
from ultralytics import YOLO

DATA_ROOT = '.' 
OUTPUT_BASE = 'ds_driveguard_temporal_roi'
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

DETECTOR = YOLO('yolov8n-pose.pt').to(DEVICE)

CAMERA_MAPPING = {
    'inner_mirror': 'ids_1',
    'a_column_co_driver': 'ids_2',
    'ceiling': 'ids_3',
    'steering_wheel': 'ids_4',
    'a_column_driver': 'ids_5'
}

CLASS_MAP = {
    'sitting_still': 'Safe', 'looking_or_moving_around (e.g. searching)': 'Safe', 
    'fastening_seat_belt': 'Safe', 'unfastening_seat_belt': 'Safe',
    'putting_on_sunglasses': 'Safe', 'taking_off_sunglasses': 'Safe',
    'drinking': 'Drink', 'opening_bottle': 'Drink', 'closing_bottle': 'Drink',
    'eating': 'Drink', 'preparing_food': 'Drink',
    'interacting_with_phone': 'Phone', 'talking_on_phone': 'Phone'
}

def get_square_box(box, img_h, img_w, padding=0.08):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw/2, y1 + bh/2
    size = max(bw, bh) * (1 + padding)
    nx1, ny1 = max(0, int(cx - size/2)), max(0, int(cy - size/2))
    nx2, ny2 = min(img_w, int(cx + size/2)), min(img_h, int(cy + size/2))
    return nx1, ny1, nx2, ny2

def get_static_roi(cap, start_idx, end_idx):
    for offset in range(min(10, end_idx - start_idx)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx + offset)
        ret, frame = cap.read()
        if not ret: break
        results = DETECTOR.predict(frame, conf=0.25, verbose=False, device=DEVICE)[0]
        if len(results.boxes) > 0:
            best_idx = int(results.boxes.conf.argmax())
            box = results.boxes.xyxy[best_idx].cpu().numpy()
            return get_square_box(box, frame.shape[0], frame.shape[1])
    return None

def process_all_views():
    if os.path.exists(OUTPUT_BASE):
        print(f"🧹 Clearing directory: {OUTPUT_BASE}")
        shutil.rmtree(OUTPUT_BASE)

    invalid_video_paths = set()

    for cam_folder in CAMERA_MAPPING.keys():
        print(f"\n🚀 Processing Camera: {cam_folder}")
        for split in ['train', 'val', 'test']:
            csv_path = os.path.join(DATA_ROOT, 'activities_3s', cam_folder, f'midlevel.chunks_90.split_0.{split}.csv')
            if not os.path.exists(csv_path): continue
            
            df = pd.read_csv(csv_path)
            df['mapped_class'] = df['activity'].map(CLASS_MAP)
            df = df.dropna(subset=['mapped_class']).copy()
            if df.empty: continue
            
            counts = df['mapped_class'].value_counts()
            target_size = int(counts.median())
            
            balanced_list = []
            for cls, group in df.groupby('mapped_class'):
                if len(group) > target_size:
                    balanced_list.append(group.sample(n=target_size, random_state=42))
                else:
                    balanced_list.append(group)
            
            df_balanced = pd.concat(balanced_list).reset_index(drop=True)
            
            for _, row in tqdm(df_balanced.iterrows(), total=len(df_balanced), desc=f"{cam_folder} | {split}"):
                video_path = os.path.join(DATA_ROOT, 'data', cam_folder, row['file_id'] + ".mp4")
                if not os.path.exists(video_path):
                    continue
                if video_path in invalid_video_paths:
                    continue

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"[WARN] Skipping unreadable video (open failed): {video_path}")
                    invalid_video_paths.add(video_path)
                    cap.release()
                    continue

                # Probe one frame so files with broken MP4 headers are skipped once and cached.
                probe_ok, _ = cap.read()
                if not probe_ok:
                    print(f"[WARN] Skipping unreadable video (frame decode failed): {video_path}")
                    invalid_video_paths.add(video_path)
                    cap.release()
                    continue

                static_box = get_static_roi(cap, int(row['frame_start']), int(row['frame_end']))
                if static_box is None:
                    cap.release()
                    continue

                seq_id = f"{cam_folder}_{row['file_id'].replace('/', '_')}_ann{row['annotation_id']}_ch{row['chunk_id']}"
                out_dir = os.path.join(OUTPUT_BASE, split, row['mapped_class'], seq_id)
                os.makedirs(out_dir, exist_ok=True)
                
                indices = np.linspace(int(row['frame_start']), int(row['frame_end']), 16, dtype=int)
                nx1, ny1, nx2, ny2 = static_box

                for i, target_idx in enumerate(indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                    ret, frame = cap.read()
                    if ret:
                        roi = frame[ny1:ny2, nx1:nx2]
                        if roi.size > 0:
                            cv2.imwrite(os.path.join(out_dir, f"frame_{i:02d}.jpg"), roi, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                cap.release()

if __name__ == "__main__":
    process_all_views()
    print(f"\nDone! Temporal ROI Dataset created: {OUTPUT_BASE}")