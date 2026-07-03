"""
Stage 1 — Build the IR frame set for object-detection labeling.

Selection is class-aware (not the natural Drive&Act activity distribution) and is
separated from extraction:
  * SELECTION builds a manifest from the CSVs alone (no video): targets ~N frames
    per intended object class via an activity->class proxy, bakes in a
    participant-grouped train/val/test split, and reports the real per-class ceiling.
  * EXTRACTION reads a manifest and writes the actual full-resolution frames.

Object class vs activity
------------------------
Drive&Act annotates *activities*, not object classes. We map activities to an
intended object class for sampling:
    phone        <- interacting_with_phone, talking_on_phone
    bottle       <- opening_bottle, closing_bottle
    food         <- eating, preparing_food
    bottle_or_cup<- drinking            (cup has NO dedicated activity)
`cup` cannot be targeted at sampling time — it only emerges from `drinking` frames
that turn out to contain a cup, determined at detection (Stage 2/3). We therefore
take the whole `drinking` pool and report cup's true count after detection.

Frozen dataset
--------------
The manifest IS the dataset definition; it builds the spec, then extract_roi_variant.py
cuts the ROI crops from it. The current frozen set is the co-driver one:
  python 1_sample_ir_frames.py --camera a_column_co_driver --per-class 800 --safe 200 \
      --freeze-to manifest_codriver_roi.csv --no-extract     # (re)build the frozen spec
  python extract_roi_variant.py                              # extract ROI crops from it
Note: this script's own --frozen extractor writes FULL frames; for the co-driver ROI
dataset use extract_roi_variant.py instead (keypoint ROI crops).
"""

import argparse
import os
import random
import sys

import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402


def _load_rows(cameras=None):
    """Collect every CSV row whose activity is in our taxonomy, tagged with obj_class.

    `cameras` restricts the source cameras (default: all in CAMERA_MAPPING)."""
    rows = []
    for cam in (cameras or C.CAMERA_MAPPING):
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
            df["obj_class"] = df["activity"].map(
                lambda a: C.SAMPLE_OBJ_MAP.get(a, "safe"))
            rows.append(df)
    if not rows:
        raise SystemExit(f"No activity CSVs found under {C.DATA_ROOT}/activities_3s")
    return pd.concat(rows, ignore_index=True)


def _take(pool, n, seed):
    """Sample up to n rows from pool, spread across cameras."""
    if len(pool) <= n:
        return pool
    sel = pool.groupby("camera", group_keys=False).sample(frac=n / len(pool), random_state=seed)
    return sel.sample(n=n, random_state=seed) if len(sel) > n else sel


def _select_class_aware(df, per_class, safe_count, seed):
    """Class-aware selection. Returns (selected_df, report dict)."""
    parts, report = [], {}
    # known-class targets (sample up to per_class each). `drink` folds the bottle +
    # drinking activities together (bottle/cup were merged into one drink class).
    for cls in C.CLASSES:  # phone, drink, food
        pool = df[df["obj_class"] == cls]
        sel = _take(pool, per_class, seed)
        parts.append(sel)
        report[cls] = {"selected": len(sel), "available": len(pool), "target": per_class}
    # small controlled negative pool
    safe = df[df["obj_class"] == "safe"]
    s = _take(safe, safe_count, seed)
    parts.append(s)
    report["safe (negatives)"] = {"selected": len(s), "available": len(safe), "target": safe_count}
    return pd.concat(parts, ignore_index=True), report


def _assign_splits(sel, ratios, seed):
    """Participant-grouped train/val/test split (no participant spans splits)."""
    counts = sel.groupby("participant_id").size()
    total = int(counts.sum())
    pids = list(counts.index)
    random.Random(seed).shuffle(pids)
    split_of, acc = {}, {"test": 0, "val": 0}
    # Pack each split up to its cap WITHOUT overshooting, so train keeps its share
    # (with only 15 participants the ratio is approximate, but train stays >= target).
    for sp in ["test", "val"]:
        cap = ratios[sp] * total
        for pid in pids:
            if pid in split_of:
                continue
            if acc[sp] + int(counts[pid]) <= cap:
                split_of[pid] = sp
                acc[sp] += int(counts[pid])
    for pid in pids:
        split_of.setdefault(pid, "train")
    return sel["participant_id"].map(split_of)


def _build_manifest(selected):
    """CSV rows -> manifest (image name + middle-frame index). No video access."""
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
            "obj_class": r["obj_class"], "split": r["split"],
            "src_split": r["src_split"], "frame": frame,
        })
    return pd.DataFrame(rows)


def _extract(manifest):
    """Write the frames listed in the manifest. Returns (#written, #missing)."""
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
        if cv2.imwrite(os.path.join(C.SAMPLED_DIR, row["image"]), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            written += 1
        else:
            missing += 1
    return written, missing


def _print_report(report, manifest):
    print("\n=== Per-class selection (ceiling report) ===")
    for cls, r in report.items():
        line = f"  {cls:30s} selected={r['selected']:5d}  available={r['available']:5d}"
        if "target" in r and r["selected"] < r["target"]:
            line += f"  ⚠ below target {r['target']} (CEILING)"
        if "note" in r:
            line += f"\n      ↳ {r['note']}"
        print(line)
    print("\n=== Split (participant-grouped 70/15/15) ===")
    print(manifest.groupby("split").size().to_string())
    print("\n=== obj_class × split ===")
    print(manifest.pivot_table(index="obj_class", columns="split",
                               values="image", aggfunc="count", fill_value=0).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=C.PER_CLASS_TARGET, help="target frames per object class")
    ap.add_argument("--safe", type=int, default=C.SAFE_COUNT, help="number of Safe negative frames")
    ap.add_argument("--camera", action="append", choices=list(C.CAMERA_MAPPING),
                    help="restrict to one or more cameras (repeatable); default: all cameras")
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--frozen", metavar="MANIFEST", help="re-extract exact frames from a committed manifest")
    ap.add_argument("--freeze-to", metavar="PATH", help="also write the manifest to PATH (committable frozen spec)")
    ap.add_argument("--no-extract", action="store_true", help="build the manifest only, don't write images")
    args = ap.parse_args()

    if args.frozen:
        manifest = pd.read_csv(args.frozen)
        print(f"Frozen mode: {len(manifest)} frames from {args.frozen}")
    else:
        df = _load_rows(args.camera).drop_duplicates(
            subset=["camera", "file_id", "annotation_id", "chunk_id"])  # 1 frame / chunk
        sel, report = _select_class_aware(df, args.per_class, args.safe, args.seed)
        sel["split"] = _assign_splits(sel, C.SPLIT_RATIOS, args.seed)
        manifest = _build_manifest(sel)
        _print_report(report, manifest)

    os.makedirs(C.SAMPLED_DIR, exist_ok=True)
    manifest.to_csv(C.MANIFEST_CSV, index=False)
    if args.freeze_to:
        manifest.to_csv(args.freeze_to, index=False)
        print(f"\nFrozen spec written: {args.freeze_to}  (commit this)")

    print(f"\nManifest: {len(manifest)} frames")
    if args.no_extract:
        print("--no-extract: manifest only, no images written.")
        return
    written, missing = _extract(manifest)
    if written == 0:
        raise SystemExit("No frames extracted — check DATA_ROOT and that data/ videos exist.")
    print(f"Extracted {written} frames to {C.SAMPLED_DIR}" + (f"  ({missing} missing)" if missing else ""))


if __name__ == "__main__":
    main()
