import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models.video import Swin3D_T_Weights, swin3d_t
from tqdm import tqdm


CLASS_NAMES = ["Safe", "Drink", "Phone"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DriveGuard16FramesDataset(Dataset):
    def __init__(self, root_dir: str, split: str, image_size: int = 224, is_train: bool = False) -> None:
        self.split_dir = Path(root_dir) / split
        self.image_size = image_size
        self.is_train = is_train
        self.samples: List[Tuple[List[Path], int]] = []
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.crop_scale = (0.8, 1.0)
        self.crop_ratio = (0.9, 1.1)
        self.jitter_brightness = 0.2
        self.jitter_contrast = 0.2
        self.jitter_saturation = 0.2
        self.jitter_hue = 0.05
        self._scan()

    def limit_fraction(self, fraction: float, seed: int) -> None:
        if len(self.samples) == 0:
            return
        if fraction >= 1.0:
            return
        if fraction <= 0.0:
            raise ValueError("limit_data must be in (0, 1].")
        keep_count = max(1, int(len(self.samples) * fraction))
        rng = random.Random(seed)
        chosen_indices = sorted(rng.sample(range(len(self.samples)), keep_count))
        self.samples = [self.samples[idx] for idx in chosen_indices]

    def _scan(self) -> None:
        if not self.split_dir.exists():
            return

        for class_name in CLASS_NAMES:
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue

            for seq_dir in class_dir.iterdir():
                if not seq_dir.is_dir():
                    continue
                frame_paths = sorted(seq_dir.glob("frame_*.jpg"))
                if len(frame_paths) == 16:
                    self.samples.append((frame_paths, CLASS_TO_IDX[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_paths, label = self.samples[idx]
        images = [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]

        if self.is_train:
            # Apply identical augmentation params over all 16 frames to preserve temporal consistency.
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                images[0],
                scale=self.crop_scale,
                ratio=self.crop_ratio,
            )
            do_hflip = random.random() < 0.5
            b_factor = 1.0 + random.uniform(-self.jitter_brightness, self.jitter_brightness)
            c_factor = 1.0 + random.uniform(-self.jitter_contrast, self.jitter_contrast)
            s_factor = 1.0 + random.uniform(-self.jitter_saturation, self.jitter_saturation)
            h_factor = random.uniform(-self.jitter_hue, self.jitter_hue)

            aug_images = []
            for image in images:
                image = TF.resized_crop(image, i, j, h, w, (self.image_size, self.image_size))
                if do_hflip:
                    image = TF.hflip(image)
                image = TF.adjust_brightness(image, b_factor)
                image = TF.adjust_contrast(image, c_factor)
                image = TF.adjust_saturation(image, s_factor)
                image = TF.adjust_hue(image, h_factor)
                aug_images.append(image)
            images = aug_images
        else:
            images = [TF.resize(image, (self.image_size, self.image_size)) for image in images]

        frames = [self.normalize(TF.to_tensor(image)) for image in images]

        # Swin3D expects input shape: (B, C, T, H, W)
        video = torch.stack(frames, dim=1)  # (C, T, H, W)
        return video, torch.tensor(label, dtype=torch.long)


def make_model(num_classes: int, use_pretrained: bool, dropout: float) -> nn.Module:
    if use_pretrained:
        model = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
    else:
        model = swin3d_t(weights=None)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(videos)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def try_batch_size(
    model: nn.Module,
    criterion: nn.Module,
    sample_video: torch.Tensor,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> bool:
    try:
        model.zero_grad(set_to_none=True)
        videos = sample_video.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(videos)
            loss = criterion(logits, labels)
        loss.backward()
        model.zero_grad(set_to_none=True)
        del videos, labels, logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if device.type == "mps":
            torch.mps.empty_cache()
        return True
    except RuntimeError as exc:
        message = str(exc).lower()
        is_memory_error = "out of memory" in message or "oom" in message
        if is_memory_error:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if device.type == "mps":
                torch.mps.empty_cache()
            return False
        raise


def auto_batch_size_probe(
    model: nn.Module,
    criterion: nn.Module,
    dataset: DriveGuard16FramesDataset,
    device: torch.device,
    use_amp: bool,
) -> None:
    if len(dataset) == 0:
        return
    sample_video, _ = dataset[0]
    candidates = [2, 4, 8]
    print("[INFO] Probing feasible batch sizes on current device...")
    feasible = []
    for candidate in candidates:
        ok = try_batch_size(model, criterion, sample_video, candidate, device, use_amp)
        print(f"[INFO] Batch size {candidate}: {'OK' if ok else 'OOM'}")
        if ok:
            feasible.append(candidate)
    if feasible:
        print(f"[INFO] Suggested max tested batch size: {max(feasible)}")
    else:
        print("[WARN] Even batch size 2 failed in probe. Lower image size may be required.")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    print(f"[INFO] Using device: {device}")

    train_ds = DriveGuard16FramesDataset(args.data_root, split="train", image_size=args.image_size, is_train=True)
    val_ds = DriveGuard16FramesDataset(args.data_root, split="val", image_size=args.image_size, is_train=False)

    train_ds.limit_fraction(args.limit_data, seed=args.seed)
    val_ds.limit_fraction(args.limit_data, seed=args.seed + 1)
    print(f"[INFO] Training samples after limit_data={args.limit_data:.2f}: {len(train_ds)}")
    print(f"[INFO] Validation samples after limit_data={args.limit_data:.2f}: {len(val_ds)}")

    if len(train_ds) == 0:
        raise RuntimeError(
            f"No training samples found in '{Path(args.data_root) / 'train'}'. "
            "Run preprocessing first and verify class folders (Safe/Drink/Phone)."
        )

    num_workers = min(os.cpu_count() or 2, args.num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = make_model(num_classes=len(CLASS_NAMES), use_pretrained=args.pretrained, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.min_lr,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if args.auto_batch_test:
        auto_batch_size_probe(model, criterion, train_ds, device, use_amp)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = Path(args.output_dir) / "best_swin3d_driveguard.pt"
    last_path = Path(args.output_dir) / "last_swin3d_driveguard.pt"

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_acc = float(checkpoint.get("val_acc", best_val_acc))
        best_val_loss = float(checkpoint.get("val_loss", best_val_loss))
        print(f"[INFO] Resumed from {resume_path} at epoch {start_epoch - 1}")
        if start_epoch > args.epochs:
            print("[INFO] Resume epoch already reaches/exceeds target epochs. Nothing to train.")
            return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for videos, labels in pbar:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(videos)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_loss = running_loss / max(total, 1)
            train_acc = running_correct / max(total, 1)
            pbar.set_postfix(loss=f"{train_loss:.4f}", acc=f"{train_acc:.4f}")

        val_loss, val_acc = evaluate(model, val_loader, criterion, device) if len(val_ds) > 0 else (0.0, 0.0)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[INFO] Epoch {epoch}: train_loss={running_loss / total:.4f} "
            f"train_acc={running_correct / total:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={current_lr:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "class_to_idx": CLASS_TO_IDX,
            "args": vars(args),
        }
        torch.save(checkpoint, last_path)

        if len(val_ds) > 0:
            improved = val_loss < (best_val_loss - args.early_stop_min_delta)
        else:
            improved = val_acc >= best_val_acc

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(checkpoint, best_path)
            print(f"[INFO] Saved new best model to {best_path}")
        elif len(val_ds) > 0:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"[INFO] Early stopping triggered after {epoch} epochs "
                    f"(no val_loss improvement for {args.early_stopping_patience} epochs)."
                )
                break

        scheduler.step()

    print(f"[INFO] Training complete. Last model: {last_path}")
    print(f"[INFO] Best model: {best_path} (val_acc={best_val_acc:.4f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Video Swin on 16-frame ROI DriveGuard dataset.")
    parser.add_argument("--data_root", type=str, default="ds_driveguard_16frames_roi.nosync")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--limit_data", type=float, default=1.0, help="Fraction of dataset to use, in (0, 1].")
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Checkpoint path to resume from (e.g. checkpoints/best_swin3d_driveguard.pt).",
    )
    parser.add_argument(
        "--auto_batch_test",
        action="store_true",
        help="Try batch sizes 2/4/8 on current device before training.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", help="Use Kinetics-400 pretrained backbone.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
