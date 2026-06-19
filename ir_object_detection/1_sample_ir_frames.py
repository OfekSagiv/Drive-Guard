"""
Stage 1 — Build the IR frame set for object-detection labeling.

Selection is separated from extraction:
  * SELECTION builds a manifest from the Drive&Act CSVs alone (no video needed):
    middle frame per chunk, Phone/Drink prioritized + ~12% Safe hard negatives.
  * EXTRACTION reads a manifest and writes the actual full-resolution frames.

Frozen dataset
--------------
The manifest IS the dataset definition. Create it once and commit it, then always
re-extract the identical frames from it — independent of the sampling RNG:

  # create the frozen spec once (fast, CSV-only, no images):
  python 1_sample_ir_frames.py --target 2000 --no-extract --freeze-to manifest_2000.csv
  #   -> commit ir_object_detection/manifest_2000.csv

  # reproduce the exact frames anytime (locally or on Colab):
  python 1_sample_ir_frames.py --frozen manifest_2000.csv

Default (no flags): sample C.TARGET_FRAMES fresh and extract them.

Output: sampled_frames/*.jpg + sampled_frames/manifest.csv
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
    """Pick rows: object chunks (capped to budget) + ~safe_frac negatives."""
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


def _build_manifest(selected):
    """Turn selected CSV rows into a manifest (image name + middle-frame index).
    Pure metadata — does NOT open any video."""
    rows = []
    for _, r in selected.iterrows():
        file_id_str = r["file_id"].split("/")[1]
        frame = (int(r["frame_start"]) + int(r["frame_end"])) // 2
        name = (f"{r['camera']}__{file_id_str}__vp{r['participant_id']}"
                f"__ann{r['annotation_id']}__ch{r['chunk_id']}{C.IMG_EXT}")
        rows.append({
            "image": name, "camera": r["camera"], "participant_id": r["participant_id"],
            "file_id": r["file_id"], "annotation_id": r["annotation_id"],
            "chunk_id": r["chunk_id"], "activity": r["activity"],
            "group": r["group"], "src_split": r["src_split"], "frame": frame,
        })
    return pd.DataFrame(rows)


def _extract(manifest):
    """Write the actual frames listed in the manifest. Returns (#written, #missing)."""
    os.makedirs(C.SAMPLED_DIR, exist_ok=True)
    bad_videos, written, missing = set(), 0, 0
    for _, row in manifest.iterrows():
        vp_folder, file_id_str = row["file_id"].split("/")
        video_path = os.path.join(C.DATA_ROOT, "data", row["camera"], vp_folder, f"{file_id_str}.mp4")
        if video_path in bad_videos or not os.path.exists(video_path):
            missing += 1; continue
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            bad_videos.add(video_path); cap.release(); missing += 1; continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            missing += 1; continue
        out_path = os.path.join(C.SAMPLED_DIR, row["image"])
        if cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            written += 1
        else:
            missing += 1
    return written, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=C.TARGET_FRAMES, help="frames to sample (sample mode)")
    ap.add_argument("--safe-frac", type=float, default=C.SAFE_NEGATIVE_FRAC)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--frozen", metavar="MANIFEST", help="re-extract the exact frames from a committed manifest")
    ap.add_argument("--freeze-to", metavar="PATH", help="also write the manifest to PATH (the committable frozen spec)")
    ap.add_argument("--no-extract", action="store_true", help="only build the manifest, don't write images")
    args = ap.parse_args()

    if args.frozen:
        manifest = pd.read_csv(args.frozen)
        print(f"Frozen mode: {len(manifest)} frames from {args.frozen}")
    else:
        df = _load_rows().drop_duplicates(
            subset=["camera", "file_id", "annotation_id", "chunk_id"])  # 1 frame / chunk
        manifest = _build_manifest(_select(df, args.target, args.safe_frac, args.seed))

    os.makedirs(C.SAMPLED_DIR, exist_ok=True)
    manifest.to_csv(C.MANIFEST_CSV, index=False)
    if args.freeze_to:
        manifest.to_csv(args.freeze_to, index=False)
        print(f"Frozen spec written: {args.freeze_to}  (commit this)")

    print(f"\nManifest: {len(manifest)} frames")
    print("By group:\n" + manifest["group"].value_counts().to_string())
    print("By camera:\n" + manifest["camera"].value_counts().to_string())

    if args.no_extract:
        print("\n--no-extract: manifest only, no images written.")
        return
    written, missing = _extract(manifest)
    if written == 0:
        raise SystemExit("No frames extracted — check DATA_ROOT and that data/ videos exist.")
    print(f"\nExtracted {written} frames to {C.SAMPLED_DIR}" + (f"  ({missing} missing)" if missing else ""))


if __name__ == "__main__":
    main()
