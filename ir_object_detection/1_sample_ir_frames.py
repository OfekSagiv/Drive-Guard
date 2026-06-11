"""
Stage 1 — Sample ~500 representative IR frames for object-detection labeling.

Strategy
--------
* Read the Drive&Act activity CSVs for all 5 cameras x {train,val,test}, reusing
  the path conventions and middle-frame selection from extract_spatial_roi_ds.py.
* Prioritize object-bearing chunks (Phone + Drink granular activities) so phones,
  bottles, cups and food are actually visible.
* Reserve ~12% of the budget for Safe frames as explicit hard negatives so the
  fine-tuned detector learns to suppress false positives.
* Keep FULL-resolution frames (no person-ROI crop): the objects are small and
  live inference runs the detector on the full frame before gating to the ROI.

Output: sampled_frames/*.jpg  +  sampled_frames/manifest.csv
Run:    python ir_object_detection/1_sample_ir_frames.py [--target 500]
"""

import argparse
import os
import sys

import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402


def _load_rows():
    """Collect every CSV row whose activity is in our taxonomy, tagged with group."""
    rows = []
    for cam in C.CAMERA_MAPPING:
        for split in C.SPLITS:
            csv_path = os.path.join(
                C.DATA_ROOT, "activities_3s", cam,
                f"midlevel.chunks_90.split_0.{split}.csv",
            )
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            df = df[df["activity"].isin(C.ALL_ACTIVITIES)].copy()
            if df.empty:
                continue
            df["camera"] = cam
            df["src_split"] = split
            df["group"] = df["activity"].map(
                lambda a: "phone" if a in C.PHONE_ACTIVITIES
                else "drink" if a in C.DRINK_ACTIVITIES else "safe"
            )
            rows.append(df)
    if not rows:
        raise SystemExit(f"No activity CSVs found under {C.DATA_ROOT}/activities_3s")
    return pd.concat(rows, ignore_index=True)


def _select(df, target, safe_frac, seed):
    """Pick rows: all object chunks (capped to budget), then ~safe_frac negatives."""
    n_safe = int(round(target * safe_frac))
    n_obj = target - n_safe

    obj = df[df["group"] != "safe"]
    safe = df[df["group"] == "safe"]

    # Object chunks: take all if within budget; else stratified sample by
    # (camera, group) so no single camera/activity dominates. groupby.sample
    # preserves the grouping columns (unlike groupby.apply in pandas 3.0).
    if len(obj) > n_obj:
        obj = obj.groupby(["camera", "group"], group_keys=False).sample(
            frac=n_obj / len(obj), random_state=seed)
        if len(obj) > n_obj:
            obj = obj.sample(n=n_obj, random_state=seed)

    safe = safe.sample(n=min(n_safe, len(safe)), random_state=seed) if len(safe) else safe
    return pd.concat([obj, safe], ignore_index=True)


def _extract(selected):
    os.makedirs(C.SAMPLED_DIR, exist_ok=True)
    bad_videos, manifest = set(), []

    for _, row in selected.iterrows():
        cam = row["camera"]
        vp_folder, file_id_str = row["file_id"].split("/")
        video_path = os.path.join(C.DATA_ROOT, "data", cam, vp_folder, f"{file_id_str}.mp4")
        if video_path in bad_videos or not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            bad_videos.add(video_path)
            cap.release()
            continue

        mid_f = (int(row["frame_start"]) + int(row["frame_end"])) // 2  # middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue

        name = (f"{cam}__{file_id_str}__vp{row['participant_id']}"
                f"__ann{row['annotation_id']}__ch{row['chunk_id']}{C.IMG_EXT}")
        out_path = os.path.join(C.SAMPLED_DIR, name)
        if not cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            continue

        manifest.append({
            "image": name, "camera": cam, "participant_id": row["participant_id"],
            "file_id": row["file_id"], "annotation_id": row["annotation_id"],
            "chunk_id": row["chunk_id"], "activity": row["activity"],
            "group": row["group"], "src_split": row["src_split"], "frame": mid_f,
        })
    return pd.DataFrame(manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=C.TARGET_FRAMES)
    ap.add_argument("--safe-frac", type=float, default=C.SAFE_NEGATIVE_FRAC)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    args = ap.parse_args()

    df = _load_rows()
    # Dedup: at most one frame per (camera, annotation, chunk).
    df = df.drop_duplicates(subset=["camera", "file_id", "annotation_id", "chunk_id"])
    selected = _select(df, args.target, args.safe_frac, args.seed)
    manifest = _extract(selected)

    if manifest.empty:
        raise SystemExit("No frames extracted — check DATA_ROOT and that data/ videos exist.")
    manifest.to_csv(C.MANIFEST_CSV, index=False)

    print(f"\nSaved {len(manifest)} frames to {C.SAMPLED_DIR}")
    print(f"Manifest: {C.MANIFEST_CSV}")
    print("\nBy group:\n" + manifest["group"].value_counts().to_string())
    print("\nBy camera:\n" + manifest["camera"].value_counts().to_string())


if __name__ == "__main__":
    main()
