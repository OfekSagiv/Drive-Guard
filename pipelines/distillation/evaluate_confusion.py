import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.video import swin3d_t


DEFAULT_CLASSES = ["Safe", "Drink", "Phone"]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # MPS can be unstable for this 3D eval path on some macOS/PyTorch versions.
        # Use CPU by default for deterministic, crash-free evaluation.
        return torch.device("cpu")
    return torch.device("cpu")


def _is_valid_dataset_root(path: Path) -> bool:
    return path.exists() and (path / "train").exists() and (path / "val").exists() and (path / "test").exists()


def resolve_data_root(data_root_arg: str) -> Path:
    raw = Path(data_root_arg).expanduser()
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent.parent  # .../gmar

    candidates = [
        raw,
        Path.cwd() / raw,
        script_dir / raw,
        workspace_root / raw,
        workspace_root / "matala" / raw.name,
    ]

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_valid_dataset_root(resolved):
            return resolved

    raise RuntimeError(
        "No valid dataset root found. Expected a folder containing train/val/test. "
        f"Tried argument '{data_root_arg}' and common locations around the project."
    )


class DriveGuard16FramesDataset(Dataset):
    def __init__(self, root_dir: str, split: str, class_names: List[str], image_size: int = 224) -> None:
        self.split_dir = Path(root_dir) / split
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples: List[Tuple[List[Path], int]] = []
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._scan()

    def _scan(self) -> None:
        if not self.split_dir.exists():
            return
        for class_name in self.class_names:
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue
            for seq_dir in class_dir.iterdir():
                if not seq_dir.is_dir():
                    continue
                frame_paths = sorted(seq_dir.glob("frame_*.jpg"))
                if len(frame_paths) == 16:
                    self.samples.append((frame_paths, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_paths, label = self.samples[idx]
        frames = []
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            frames.append(self.transform(image))
        video = torch.stack(frames, dim=1)  # (C, T, H, W)
        return video, torch.tensor(label, dtype=torch.long)


def build_model(num_classes: int, dropout: float) -> nn.Module:
    model = swin3d_t(weights=None)
    in_features = model.head[1].in_features if isinstance(model.head, nn.Sequential) else model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path | None) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Test Set)",
    )

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"[INFO] Saved confusion matrix plot to: {save_path}")
    plt.show()


def update_confusion_matrix(cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Safely update confusion matrix and skip invalid class indices.
    Returns the number of skipped (true, pred) pairs.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    num_classes = cm.shape[0]
    skipped = 0
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
        else:
            skipped += 1
    return skipped


def compute_classification_metrics_from_cm(
    cm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)

    supports = cm.sum(axis=1).astype(np.float64)
    predicted_totals = cm.sum(axis=0).astype(np.float64)

    for idx in range(num_classes):
        tp = float(cm[idx, idx])
        fp = float(predicted_totals[idx] - tp)
        fn = float(supports[idx] - tp)

        denom_precision = tp + fp
        denom_recall = tp + fn
        precision[idx] = tp / denom_precision if denom_precision > 0 else 0.0
        recall[idx] = tp / denom_recall if denom_recall > 0 else 0.0

        denom_f1 = precision[idx] + recall[idx]
        f1[idx] = (2.0 * precision[idx] * recall[idx] / denom_f1) if denom_f1 > 0 else 0.0

    macro_f1 = float(f1.mean()) if num_classes > 0 else 0.0
    total_support = float(supports.sum())
    weighted_f1 = float((f1 * supports).sum() / total_support) if total_support > 0 else 0.0
    return precision, recall, f1, macro_f1, weighted_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model on test set and print confusion matrix details.")
    parser.add_argument("--data_root", type=str, default="ds_driveguard_16frames_roi.nosync")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_swin3d_driveguard.pt")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50,
        help="Print progress every N batches during evaluation.",
    )
    parser.add_argument("--save_plot", type=str, default="checkpoints/confusion_matrix_test.png")
    parser.add_argument("--no_plot", action="store_true", help="Disable interactive confusion matrix display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"[INFO] Using device: {device}")
    data_root = resolve_data_root(args.data_root)
    print(f"[INFO] Dataset root: {data_root}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    class_to_idx: Dict[str, int] = checkpoint.get("class_to_idx", {name: i for i, name in enumerate(DEFAULT_CLASSES)})
    idx_to_class = [""] * len(class_to_idx)
    for class_name, idx in class_to_idx.items():
        idx_to_class[idx] = class_name

    train_args = checkpoint.get("args", {})
    dropout = float(train_args.get("dropout", 0.3))
    model = build_model(num_classes=len(idx_to_class), dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = DriveGuard16FramesDataset(
        root_dir=str(data_root),
        split="test",
        class_names=idx_to_class,
        image_size=args.image_size,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No valid 16-frame samples found in: {data_root / 'test'}")

    requested_workers = args.num_workers
    if device.type == "mps" and requested_workers > 0:
        print("[INFO] MPS detected: using num_workers=0 for stability.")
        requested_workers = 0
    num_workers = min(requested_workers, (torch.get_num_threads() or 1))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    total_batches = len(loader)
    print(
        f"[INFO] Evaluation setup: samples={len(dataset)}, batch_size={args.batch_size}, "
        f"batches={total_batches}, num_workers={num_workers}"
    )
    if args.progress_every <= 0:
        args.progress_every = 1

    cm = np.zeros((len(idx_to_class), len(idx_to_class)), dtype=np.int64)
    total = 0
    correct = 0
    skipped_pairs = 0

    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(loader, start=1):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(videos)
            preds = logits.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            y_true = labels.detach().to("cpu", dtype=torch.long).numpy()
            y_pred = preds.detach().to("cpu", dtype=torch.long).numpy()
            skipped_pairs += update_confusion_matrix(cm, y_true, y_pred)

            if batch_idx == 1 or (batch_idx % args.progress_every == 0) or batch_idx == total_batches:
                running_acc = correct / max(total, 1)
                print(
                    f"[INFO] Progress {batch_idx}/{total_batches} batches "
                    f"({total}/{len(dataset)} samples), running_acc={running_acc:.4f}"
                )

    accuracy = correct / max(total, 1)
    print(f"[RESULT] Test samples: {total}")
    print(f"[RESULT] Test accuracy: {accuracy:.4f}")
    print("[RESULT] Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    if skipped_pairs > 0:
        print(f"[WARN] Skipped {skipped_pairs} invalid label/pred pairs while building confusion matrix.")
    precision, recall, f1, macro_f1, weighted_f1 = compute_classification_metrics_from_cm(cm)
    print("[RESULT] Per-class metrics:")
    for idx, class_name in enumerate(idx_to_class):
        support = int(cm[idx, :].sum())
        print(
            f"  - {class_name}: "
            f"precision={precision[idx]:.4f}, recall={recall[idx]:.4f}, f1={f1[idx]:.4f}, support={support}"
        )
    print(f"[RESULT] Macro F1: {macro_f1:.4f}")
    print(f"[RESULT] Weighted F1: {weighted_f1:.4f}")

    if "Drink" in class_to_idx and "Safe" in class_to_idx:
        drink_idx = class_to_idx["Drink"]
        safe_idx = class_to_idx["Safe"]
        drink_total = int(cm[drink_idx, :].sum())
        drink_as_safe = int(cm[drink_idx, safe_idx])
        ratio = (drink_as_safe / drink_total * 100.0) if drink_total > 0 else 0.0
        print(f"[FOCUS] Drink total in test: {drink_total}")
        print(f"[FOCUS] Drink -> Safe misclassifications: {drink_as_safe}")
        print(f"[FOCUS] Drink -> Safe rate: {ratio:.2f}%")
    else:
        print("[WARN] Could not compute Drink -> Safe metric because one of the classes is missing.")

    if not args.no_plot:
        save_path = Path(args.save_plot) if args.save_plot else None
        plot_confusion_matrix(cm, idx_to_class, save_path)


if __name__ == "__main__":
    main()