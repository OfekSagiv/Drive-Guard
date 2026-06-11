"""
Stage 4 — Assemble the final YOLO dataset + QC report.

* Builds the Ultralytics layout (images/{train,val}, labels/{train,val}, data.yaml).
* Splits train/val by participant_id (group split, NOT per-image random) so the
  same person/session never spans both sets — prevents leakage.
* Emits qc_stats.json + qc_report.md (class distribution, boxes/image, negatives,
  per-camera/per-class breakdown, decision-mix from the review log).

The dataset is the auto-pipeline's *proposal*. Do NOT train on it before the
human-validation gate in the README is satisfied.

Run: python ir_object_detection/4_build_dataset.py [--val-frac 0.15]
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402


def _assign_splits(manifest, val_frac, seed):
    """Group by participant_id; greedily fill val to ~val_frac of images."""
    counts = manifest.groupby("participant_id").size().sort_values(ascending=False)
    total = len(manifest)
    target_val = int(round(total * val_frac))
    rng = list(counts.index)
    # deterministic: largest-first fill keeps the split stable
    val_pids, acc = set(), 0
    for pid in rng:
        if acc >= target_val:
            break
        val_pids.add(pid)
        acc += counts[pid]
    return {pid: ("val" if pid in val_pids else "train") for pid in counts.index}


def _read_label(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [ln.split() for ln in f.read().splitlines() if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=C.VAL_FRACTION)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    args = ap.parse_args()

    manifest = pd.read_csv(C.MANIFEST_CSV)
    split_of_pid = _assign_splits(manifest, args.val_frac, args.seed)

    if os.path.exists(C.DATASET_DIR):
        shutil.rmtree(C.DATASET_DIR)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(C.DATASET_DIR, sub), exist_ok=True)

    # decision mix from review log (optional)
    decisions = Counter()
    if os.path.exists(C.REVIEW_JSONL):
        with open(C.REVIEW_JSONL) as f:
            for line in f:
                decisions[json.loads(line).get("decision", "?")] += 1

    per_split = defaultdict(lambda: {"images": 0, "negatives": 0, "boxes": 0,
                                     "class_counts": Counter(), "camera_counts": Counter()})
    box_sizes = []

    for _, row in manifest.iterrows():
        name = row["image"]
        split = split_of_pid.get(row["participant_id"], "train")
        label_src = os.path.join(C.REFINED_LABELS_DIR, name.replace(C.IMG_EXT, ".txt"))
        labels = _read_label(label_src)

        shutil.copy(os.path.join(C.SAMPLED_DIR, name),
                    os.path.join(C.DATASET_DIR, "images", split, name))
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
            box_sizes.append(float(l[3]) * float(l[4]))  # normalized area

    # data.yaml
    with open(os.path.join(C.DATASET_DIR, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(C.DATASET_DIR)}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write(f"nc: {len(C.CLASSES)}\n")
        f.write(f"names: {C.CLASSES}\n")

    # qc stats
    stats = {
        "classes": C.CLASSES,
        "val_fraction_target": args.val_frac,
        "splits": {sp: {
            "images": d["images"], "negatives": d["negatives"], "boxes": d["boxes"],
            "class_counts": dict(d["class_counts"]), "camera_counts": dict(d["camera_counts"]),
        } for sp, d in per_split.items()},
        "decision_mix": dict(decisions),
        "box_area_norm": {
            "n": len(box_sizes),
            "mean": (sum(box_sizes) / len(box_sizes)) if box_sizes else 0,
            "median": (sorted(box_sizes)[len(box_sizes) // 2]) if box_sizes else 0,
        },
        "participants": {"train": sorted(p for p, s in split_of_pid.items() if s == "train"),
                         "val": sorted(p for p, s in split_of_pid.items() if s == "val")},
    }
    with open(C.QC_STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # qc report (markdown)
    lines = ["# IR Object-Detection Dataset — QC Report", "",
             f"Classes: `{C.CLASSES}`  •  val grouped by participant_id (no leakage)", ""]
    for sp in ["train", "val"]:
        d = per_split[sp]
        lines += [f"## {sp}",
                  f"- images: {d['images']}  (negatives / no-object: {d['negatives']})",
                  f"- boxes: {d['boxes']}",
                  f"- class counts: {dict(d['class_counts'])}",
                  f"- camera counts: {dict(d['camera_counts'])}", ""]
    lines += ["## Decision mix (from review log)", f"- {dict(decisions)}", "",
              "## Box size (normalized area)",
              f"- n={stats['box_area_norm']['n']}, mean={stats['box_area_norm']['mean']:.5f}, "
              f"median={stats['box_area_norm']['median']:.5f}", "",
              "## Participant split",
              f"- train pids: {stats['participants']['train']}",
              f"- val pids:   {stats['participants']['val']}", "",
              "## Human-validation gate (REQUIRED before training)",
              "1. Review 100% of `refined/flagged/`.",
              "2. Audit a random 10% of auto-accepted labels.",
              "3. Build a ~50-image gold set; measure pipeline precision/recall.",
              "4. Proceed only if gold precision >= 0.90 and recall >= 0.80.",
              "5. Visual audit tool: FiftyOne (`pip install fiftyone`)."]
    with open(C.QC_REPORT_MD, "w") as f:
        f.write("\n".join(lines))

    print(f"Dataset: {C.DATASET_DIR}")
    print(f"  train: {per_split['train']['images']} imgs / {per_split['train']['boxes']} boxes")
    print(f"  val:   {per_split['val']['images']} imgs / {per_split['val']['boxes']} boxes")
    print(f"data.yaml, qc_stats.json, qc_report.md written.")


if __name__ == "__main__":
    main()
