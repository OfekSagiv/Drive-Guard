"""
Build a leakage-free gold set for measuring auto-label quality.

* Samples N images (default 150) from the frozen manifest, stratified across
  4 groups: phone, food, bottle/cup-pool (bottle + bottle_or_cup), safe.
* Leakage-free: draws ONLY from the held-out splits (default val + test), so the
  gold set never overlaps the train participants a detector would learn from.
* Extracts those exact frames straight from the raw videos into gold/images/,
  writes empty YOLO label stubs into gold/labels/ for you to annotate by hand,
  and a gold_manifest.csv tracking each image (activity, camera, participant, split).

Run: python ir_object_detection/make_gold_set.py [--n 150] [--splits val,test] [--seed 42]
"""

import argparse
import os
import sys

import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

GOLD_DIR = os.path.join(C.PKG_DIR, "gold")
GOLD_IMAGES = os.path.join(GOLD_DIR, "images")
GOLD_LABELS = os.path.join(GOLD_DIR, "labels")
GOLD_MANIFEST = os.path.join(GOLD_DIR, "gold_manifest.csv")

# 4 stratification groups -> which obj_class values feed them
STRATA = {
    "phone": ["phone"],
    "food": ["food"],
    "bottle_cup_pool": ["bottle", "bottle_or_cup"],
    "safe": ["safe"],
}


def _allocate(avail, n):
    """Spread n across strata: equal share, capped by availability, redistribute shortfall."""
    keys = list(avail)
    alloc = {k: 0 for k in keys}
    remaining = n
    # round-robin give 1 at a time to strata with spare capacity (fair + respects caps)
    while remaining > 0 and any(alloc[k] < avail[k] for k in keys):
        for k in keys:
            if remaining == 0:
                break
            if alloc[k] < avail[k]:
                alloc[k] += 1
                remaining -= 1
    return alloc


def _extract_frame(row):
    vp_folder, file_id_str = row["file_id"].split("/")
    vpath = os.path.join(C.DATA_ROOT, "data", row["camera"], vp_folder, f"{file_id_str}.mp4")
    if not os.path.exists(vpath):
        return None
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        cap.release(); return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=C.FROZEN_MANIFEST)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--splits", default="val,test", help="held-out splits to draw from")
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    splits = [s.strip() for s in args.splits.split(",")]
    df = df[df["split"].isin(splits)].copy()
    if df.empty:
        raise SystemExit(f"No rows for splits {splits} in {args.manifest}")

    df["stratum"] = df["obj_class"].map(
        {v: k for k, vals in STRATA.items() for v in vals})

    avail = {k: int((df["stratum"] == k).sum()) for k in STRATA}
    alloc = _allocate(avail, args.n)

    print(f"Drawing {args.n} gold images from splits {splits} (leakage-free):")
    picks = []
    for k in STRATA:
        pool = df[df["stratum"] == k]
        take = min(alloc[k], len(pool))
        picks.append(pool.sample(n=take, random_state=args.seed))
        flag = "" if take == alloc[k] else "  ⚠ capped by availability"
        print(f"  {k:16s} {take:4d} / requested {alloc[k]:4d}  (available {avail[k]}){flag}")
    gold = pd.concat(picks, ignore_index=True)

    os.makedirs(GOLD_IMAGES, exist_ok=True)
    os.makedirs(GOLD_LABELS, exist_ok=True)
    written, missing, rows = 0, 0, []
    for _, r in gold.iterrows():
        frame = _extract_frame(r)
        if frame is None:
            missing += 1; continue
        cv2.imwrite(os.path.join(GOLD_IMAGES, r["image"]), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        open(os.path.join(GOLD_LABELS, r["image"].replace(C.IMG_EXT, ".txt")), "w").close()
        written += 1
        rows.append({"image": r["image"], "activity": r["activity"], "camera": r["camera"],
                     "participant_id": r["participant_id"], "split": r["split"],
                     "obj_class": r["obj_class"], "stratum": r["stratum"]})

    pd.DataFrame(rows).to_csv(GOLD_MANIFEST, index=False)
    print(f"\nWrote {written} gold images -> {GOLD_IMAGES}" + (f"  ({missing} missing)" if missing else ""))
    print(f"Empty label stubs -> {GOLD_LABELS}")
    print(f"Tracking manifest  -> {GOLD_MANIFEST}")
    gdf = pd.DataFrame(rows)
    if not gdf.empty:
        print("\nParticipants in gold set (held-out only):",
              sorted(gdf["participant_id"].unique().tolist()))
        print("By stratum:\n" + gdf["stratum"].value_counts().to_string())


if __name__ == "__main__":
    main()
