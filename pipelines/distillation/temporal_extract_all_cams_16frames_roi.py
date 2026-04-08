import os
import argparse
import cv2
import pandas as pd
import numpy as np
import torch
import shutil
from tqdm import tqdm
from ultralytics import YOLO

DATA_ROOT = '.'
OUTPUT_BASE = 'ds_driveguard_16frames_roi.nosync'
DRY_RUN = False
DATA_DIR_NAMES = ['data', 'data 2', 'data 3', 'data 4', 'data 5', 'data 6']
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
    cx, cy = x1 + bw / 2, y1 + bh / 2
    size = max(bw, bh) * (1 + padding)
    nx1, ny1 = max(0, int(cx - size / 2)), max(0, int(cy - size / 2))
    nx2, ny2 = min(img_w, int(cx + size / 2)), min(img_h, int(cy + size / 2))
    return nx1, ny1, nx2, ny2


def get_static_roi(cap, start_idx, end_idx):
    for offset in range(min(10, end_idx - start_idx)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx + offset)
        ret, frame = cap.read()
        if not ret:
            break
        results = DETECTOR.predict(frame, conf=0.25, verbose=False, device=DEVICE)[0]
        if len(results.boxes) > 0:
            best_idx = int(results.boxes.conf.argmax())
            box = results.boxes.xyxy[best_idx].cpu().numpy()
            return get_square_box(box, frame.shape[0], frame.shape[1])
    return None


def discover_cameras():
    activities_root = os.path.join(DATA_ROOT, 'activities_3s')
    if not os.path.exists(activities_root):
        print(f"[ERROR] Missing activities root: {activities_root}")
        return []
    cameras = sorted(
        [
            name
            for name in os.listdir(activities_root)
            if os.path.isdir(os.path.join(activities_root, name))
        ]
    )
    if cameras:
        print(f"[INFO] Cameras discovered from CSV folders: {', '.join(cameras)}")
    return cameras


def list_existing_data_roots():
    roots = []
    for dir_name in DATA_DIR_NAMES:
        candidate = dir_name if os.path.isabs(dir_name) else os.path.join(DATA_ROOT, dir_name)
        if os.path.exists(candidate) and os.path.isdir(candidate):
            roots.append(candidate)
    if not roots:
        print("[ERROR] No data directories found. Checked: " + ", ".join(DATA_DIR_NAMES))
    return roots


def list_data_camera_folders(data_roots):
    camera_folders = set()
    for data_root in data_roots:
        for name in os.listdir(data_root):
            path = os.path.join(data_root, name)
            if os.path.isdir(path):
                camera_folders.add(name)
    return sorted(camera_folders)


def resolve_video_path(cam_folder, file_id, data_roots, data_camera_folders):
    relative_file = str(file_id) + ".mp4"
    for data_root in data_roots:
        candidates = [
            os.path.join(data_root, cam_folder, relative_file),
            os.path.join(data_root, relative_file),
        ]
        for data_cam in data_camera_folders:
            candidates.append(os.path.join(data_root, data_cam, relative_file))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    return None


def process_all_views():
    if os.path.exists(OUTPUT_BASE):
        print(f"Clearing directory: {OUTPUT_BASE}")
        shutil.rmtree(OUTPUT_BASE)

    invalid_video_paths = set()
    cameras = discover_cameras()
    data_roots = list_existing_data_roots()
    data_camera_folders = list_data_camera_folders(data_roots) if data_roots else []
    print(f"[INFO] Data directories used for lookup: {', '.join(data_roots) if data_roots else 'none'}")
    print(f"[INFO] Camera folders available under data dirs: {', '.join(data_camera_folders) if data_camera_folders else 'none'}")
    print(
        "[INFO] Expected camera folders from mapping: "
        + ", ".join(CAMERA_MAPPING.keys())
    )

    for cam_folder in cameras:
        print(f"\nProcessing Camera: {cam_folder}")
        for split in ['train', 'val', 'test']:
            stats = {
                'csv_missing': 0,
                'rows_total': 0,
                'rows_after_label_map': 0,
                'rows_balanced': 0,
                'missing_video': 0,
                'rows_with_video': 0,
                'invalid_video_cached': 0,
                'open_failed': 0,
                'decode_failed': 0,
                'roi_not_found': 0,
                'sequences_saved': 0,
            }
            csv_path = os.path.join(DATA_ROOT, 'activities_3s', cam_folder, f'midlevel.chunks_90.split_0.{split}.csv')
            if not os.path.exists(csv_path):
                stats['csv_missing'] = 1
                print(f"[WARN] Missing CSV for {cam_folder}/{split}: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            stats['rows_total'] = len(df)
            df['mapped_class'] = df['activity'].map(CLASS_MAP)
            df = df.dropna(subset=['mapped_class']).copy()
            stats['rows_after_label_map'] = len(df)
            if df.empty:
                print(f"[WARN] No rows after class mapping for {cam_folder}/{split}")
                continue

            counts = df['mapped_class'].value_counts()
            target_size = int(counts.median())

            balanced_list = []
            for cls, group in df.groupby('mapped_class'):
                if len(group) > target_size:
                    balanced_list.append(group.sample(n=target_size, random_state=42))
                else:
                    balanced_list.append(group)

            df_balanced = pd.concat(balanced_list).reset_index(drop=True)
            stats['rows_balanced'] = len(df_balanced)

            for _, row in tqdm(df_balanced.iterrows(), total=len(df_balanced), desc=f"{cam_folder} | {split}"):
                video_path = resolve_video_path(cam_folder, row['file_id'], data_roots, data_camera_folders)
                if video_path is None:
                    stats['missing_video'] += 1
                    continue
                stats['rows_with_video'] += 1
                if DRY_RUN:
                    continue
                if video_path in invalid_video_paths:
                    stats['invalid_video_cached'] += 1
                    continue

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"[WARN] Skipping unreadable video (open failed): {video_path}")
                    stats['open_failed'] += 1
                    invalid_video_paths.add(video_path)
                    cap.release()
                    continue

                # Probe one frame so files with broken MP4 headers are skipped once and cached.
                probe_ok, _ = cap.read()
                if not probe_ok:
                    print(f"[WARN] Skipping unreadable video (frame decode failed): {video_path}")
                    stats['decode_failed'] += 1
                    invalid_video_paths.add(video_path)
                    cap.release()
                    continue

                static_box = get_static_roi(cap, int(row['frame_start']), int(row['frame_end']))
                if static_box is None:
                    stats['roi_not_found'] += 1
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
                stats['sequences_saved'] += 1
                cap.release()

            print(
                f"[SUMMARY] {cam_folder}/{split}: "
                f"rows={stats['rows_total']}, mapped={stats['rows_after_label_map']}, balanced={stats['rows_balanced']}, "
                f"saved={stats['sequences_saved']}, rows_with_video={stats['rows_with_video']}, missing_video={stats['missing_video']}, "
                f"roi_not_found={stats['roi_not_found']}, open_failed={stats['open_failed']}, "
                f"decode_failed={stats['decode_failed']}, invalid_cached={stats['invalid_video_cached']}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 16-frame ROI sequences for multi-camera DriveGuard.")
    parser.add_argument("--data_root", type=str, default=".", help="Root containing 'activities_3s' and 'data' folders.")
    parser.add_argument(
        "--data_dirs",
        nargs="+",
        default=DATA_DIR_NAMES,
        help="List of data directories to search for videos (e.g. data 'data 2' 'data 3').",
    )
    parser.add_argument("--output_base", type=str, default=OUTPUT_BASE, help="Output dataset folder.")
    parser.add_argument("--dry_run", action="store_true", help="Only validate CSV/video coverage without writing frames.")
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    DATA_DIR_NAMES = args.data_dirs
    OUTPUT_BASE = args.output_base
    DRY_RUN = args.dry_run
    process_all_views()
    print(f"\nDone. Multi-view ROI Dataset created: {OUTPUT_BASE}")
