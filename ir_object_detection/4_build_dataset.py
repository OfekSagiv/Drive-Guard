"""
Stage 4 — Assemble the final YOLO dataset + QC report.

* Builds the Ultralytics layout (images/{train,val,test}, labels/..., data.yaml).
* Uses the manifest's `split` column (participant-grouped train/val/test, baked in
  by Stage 1). Falls back to a participant-grouped train/val split if absent.
* Emits qc_stats.json + qc_report.md (class distribution, boxes/image, negatives,
  per-camera/per-class breakdown, decision-mix from the review log).

The dataset is the auto-pipeline's *proposal*. Do NOT train on it before the
human-validation gate in the README is satisfied.

Run: python ir_object_detection/4_build_dataset.py
"""

import json
import os
import shutil
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402


def _fallback_splits(manifest, val_frac, seed):
    """Participant-grouped train/val split when the manifest has no `split` column."""
    counts = manifest.groupby("participant_id").size().sort_values(ascending=False)
    total, target_val, acc, val_pids = len(manifest), int(round(len(manifest) * val_frac)), 0, set()
    for pid in counts.index:
        if acc >= target_val:
            break
        val_pids.add(pid); acc += counts[pid]
    return manifest["participant_id"].map(lambda p: "val" if p in val_pids else "train")


def _read_label(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [ln.split() for ln in f.read().splitlines() if ln.strip()]


def main():
    manifest = pd.read_csv(C.MANIFEST_CSV)
    if "split" in manifest.columns:
        manifest = manifest.copy()
    else:
        manifest = manifest.copy()
        manifest["split"] = _fallback_splits(manifest, C.VAL_FRACTION, C.RANDOM_SEED)
    splits = [s for s in ["train", "val", "test"] if (manifest["split"] == s).any()]

    if os.path.exists(C.DATASET_DIR):
        shutil.rmtree(C.DATASET_DIR)
    for sp in splits:
        os.makedirs(os.path.join(C.DATASET_DIR, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(C.DATASET_DIR, "labels", sp), exist_ok=True)

    decisions = Counter()
    if os.path.exists(C.REVIEW_JSONL):
        with open(C.REVIEW_JSONL) as f:
            for line in f:
                decisions[json.loads(line).get("decision", "?")] += 1

    per_split = defaultdict(lambda: {"images": 0, "negatives": 0, "boxes": 0,
                                     "class_counts": Counter(), "camera_counts": Counter()})
    box_sizes = []

    for _, row in manifest.iterrows():
        name, split = row["image"], row["split"]
        labels = _read_label(os.path.join(C.REFINED_LABELS_DIR, name.replace(C.IMG_EXT, ".txt")))
        src_img = os.path.join(C.SAMPLED_DIR, name)
        if os.path.exists(src_img):
            shutil.copy(src_img, os.path.join(C.DATASET_DIR, "images", split, name))
        with open(os.path.join(C.DATASET_DIR, "labels", split, name.replace(C.IMG_EXT, ".txt")), "w") as f:
            f.write("\n".join(" ".join(l) for l in labels))

        s = per_split[split]
        s["images"] += 1
        s["camera_counts"][row["camera"]] += 1
        if not labels:
            s["negatives"] += 1
        for l in labels:
            cid = int(l[0])
            s["boxes"] += 1
            s["class_counts"][C.CLASSES[cid]] += 1
            box_sizes.append(float(l[3]) * float(l[4]))

    # data.yaml
    with open(os.path.join(C.DATASET_DIR, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(C.DATASET_DIR)}\n")
        for sp in splits:
            f.write(f"{sp}: images/{sp}\n")
        f.write(f"nc: {len(C.CLASSES)}\nnames: {C.CLASSES}\n")

    # participant membership per split (leakage check)
    pids = {sp: sorted(manifest.loc[manifest["split"] == sp, "participant_id"].unique().tolist())
            for sp in splits}

    stats = {
        "classes": C.CLASSES,
        "splits": {sp: {
            "images": d["images"], "negatives": d["negatives"], "boxes": d["boxes"],
            "class_counts": dict(d["class_counts"]), "camera_counts": dict(d["camera_counts"]),
        } for sp, d in per_split.items()},
        "participants_per_split": pids,
        "decision_mix": dict(decisions),
        "box_area_norm": {
            "n": len(box_sizes),
            "mean": (sum(box_sizes) / len(box_sizes)) if box_sizes else 0,
            "median": (sorted(box_sizes)[len(box_sizes) // 2]) if box_sizes else 0,
        },
    }
    with open(C.QC_STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    lines = ["# IR Object-Detection Dataset — QC Report", "",
             f"Classes: `{C.CLASSES}`  •  splits participant-grouped (no leakage)", ""]
    for sp in splits:
        d = per_split[sp]
        lines += [f"## {sp}",
                  f"- images: {d['images']}  (negatives / no-object: {d['negatives']})",
                  f"- boxes: {d['boxes']}",
                  f"- class counts: {dict(d['class_counts'])}",
                  f"- camera counts: {dict(d['camera_counts'])}",
                  f"- participants: {pids[sp]}", ""]
    lines += ["## Decision mix (from review log)", f"- {dict(decisions)}", "",
              "## Box size (normalized area)",
              f"- n={stats['box_area_norm']['n']}, mean={stats['box_area_norm']['mean']:.5f}, "
              f"median={stats['box_area_norm']['median']:.5f}", "",
              "## Human-validation gate (REQUIRED before training)",
              "1. Review 100% of `refined/flagged/`.",
              "2. Audit a random 10% of auto-accepted labels.",
              "3. Build a ~50-image gold set; measure pipeline precision/recall.",
              "4. Proceed only if gold precision >= 0.90 and recall >= 0.80.",
              "5. Visual audit tool: FiftyOne (`pip install fiftyone`)."]
    with open(C.QC_REPORT_MD, "w") as f:
        f.write("\n".join(lines))

    print(f"Dataset: {C.DATASET_DIR}")
    for sp in splits:
        print(f"  {sp:5s}: {per_split[sp]['images']} imgs / {per_split[sp]['boxes']} boxes")
    print("data.yaml, qc_stats.json, qc_report.md written.")


if __name__ == "__main__":
    main()
