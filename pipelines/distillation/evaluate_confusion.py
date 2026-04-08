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
        return torch.device("mps")
    return torch.device("cpu")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model on test set and print confusion matrix details.")
    parser.add_argument("--data_root", type=str, default="ds_driveguard_16frames_roi.nosync")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_swin3d_driveguard.pt")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_plot", type=str, default="checkpoints/confusion_matrix_test.png")
    parser.add_argument("--no_plot", action="store_true", help="Disable interactive confusion matrix display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"[INFO] Using device: {device}")

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
        root_dir=args.data_root,
        split="test",
        class_names=idx_to_class,
        image_size=args.image_size,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No valid 16-frame samples found in: {Path(args.data_root) / 'test'}")

    num_workers = min(args.num_workers, (torch.get_num_threads() or 1))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    cm = np.zeros((len(idx_to_class), len(idx_to_class)), dtype=np.int64)
    total = 0
    correct = 0

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(videos)
            preds = logits.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            y_true = labels.cpu().numpy()
            y_pred = preds.cpu().numpy()
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1

    accuracy = correct / max(total, 1)
    print(f"[RESULT] Test samples: {total}")
    print(f"[RESULT] Test accuracy: {accuracy:.4f}")
    print("[RESULT] Confusion Matrix (rows=true, cols=pred):")
    print(cm)

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
