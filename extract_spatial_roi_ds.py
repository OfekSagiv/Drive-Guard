import os
import cv2
import pandas as pd
import torch
import shutil
from tqdm import tqdm
from ultralytics import YOLO

DATA_ROOT = '.'
OUTPUT_BASE = 'ds_driveguard_spatial_roi'
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

def get_square_box(box, img_h, img_w, padding=0.12):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    size = max(bw, bh) * (1 + padding)
    nx1, ny1 = max(0, int(cx - size / 2)), max(0, int(cy - size / 2))
    nx2, ny2 = min(img_w, int(cx + size / 2)), min(img_h, int(cy + size / 2))
    return nx1, ny1, nx2, ny2

def get_roi_box(cap, start_idx, end_idx):
    """Try up to 10 frames from chunk start to find a YOLO detection.
    Returns (x1, y1, x2, y2) if found, else None (caller falls back to full frame)."""
    for offset in range(min(10, end_idx - start_idx)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx + offset)
        ret, frame = cap.read()
        if not ret:
            break
        results = DETECTOR.predict(frame, conf=0.15, verbose=False, device=DEVICE)[0]
        if len(results.boxes) > 0:
            best_idx = int(results.boxes.conf.argmax())
            box = results.boxes.xyxy[best_idx].cpu().numpy()
            return get_square_box(box, frame.shape[0], frame.shape[1])
    return None

def crop_roi(frame, box):
    """Apply ROI box to frame. Returns ROI crop, or full frame if box is None."""
    if box is None:
        return frame
    nx1, ny1, nx2, ny2 = box
    roi = frame[ny1:ny2, nx1:nx2]
    return roi if roi.size > 0 else frame

def process_all_views():
    if os.path.exists(OUTPUT_BASE):
        print(f"Clearing directory: {OUTPUT_BASE}")
        shutil.rmtree(OUTPUT_BASE)

    invalid_video_paths = set()

    for cam_folder in CAMERA_MAPPING.keys():
        print(f"\nProcessing Camera: {cam_folder}")

        for split in ['train', 'val', 'test']:
            csv_path = os.path.join(DATA_ROOT, 'activities_3s', cam_folder, f'midlevel.chunks_90.split_0.{split}.csv')
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            df = df[df['activity'].isin(CLASS_MAP.keys())].copy()
            if df.empty:
                continue

            saved_count = 0

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{cam_folder} | {split}"):
                vp_folder, file_id_str = row['file_id'].split('/')
                video_path = os.path.join(DATA_ROOT, 'data', cam_folder, vp_folder, f"{file_id_str}.mp4")

                if not os.path.exists(video_path):
                    continue
                if video_path in invalid_video_paths:
                    continue

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    invalid_video_paths.add(video_path)
                    cap.release()
                    continue

                probe_ok, _ = cap.read()
                if not probe_ok:
                    invalid_video_paths.add(video_path)
                    cap.release()
                    continue

                activity_name = CLASS_MAP[row['activity']]
                start_f = int(row['frame_start'])
                end_f = int(row['frame_end'])
                mid_f = (start_f + end_f) // 2

                # 1 frame for Safe (majority), 3 frames for Drink/Phone (minority boost)
                indices = [mid_f] if activity_name == 'Safe' else [start_f + 5, mid_f, end_f - 5]

                roi_box = get_roi_box(cap, start_f, end_f)

                out_dir = os.path.join(OUTPUT_BASE, split, activity_name, cam_folder)
                os.makedirs(out_dir, exist_ok=True)

                ann_id = row['annotation_id']
                chunk_id = row['chunk_id']

                for i, target_idx in enumerate(indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                    ret, frame = cap.read()
                    if ret:
                        img = crop_roi(frame, roi_box)
                        file_name = f"{file_id_str}__ann_{ann_id}__chunk_{chunk_id}__f_{i}.jpg"
                        out_path = os.path.join(out_dir, file_name)
                        if cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                            saved_count += 1

                cap.release()

            print(f"  {cam_folder} {split}: {saved_count} saved")

if __name__ == "__main__":
    process_all_views()
    print(f"\nDone! Spatial ROI Dataset created: {OUTPUT_BASE}")
