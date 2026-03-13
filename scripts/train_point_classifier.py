"""
Point Classifier Training Script
训练 wide_top 相机图像的毛巾起始点位分类器

用法: /venv/mult-act/bin/python3 train_point_classifier.py
"""

import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path

from point_classifier_model import PointClassifier


class PointImageDataset(Dataset):
    """从提取的图像文件夹加载数据，label 为 point 编号 (0-9)"""

    def __init__(self, data_dir, transform=None):
        self.samples = []  # (image_path, label)
        self.transform = transform

        # point1 -> 0, point2 -> 1, ..., point10 -> 9
        data_dir = Path(data_dir)
        point_dirs = sorted(
            [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("point")],
            key=lambda x: int(x.name.replace("point", ""))
        )

        self.class_names = [d.name for d in point_dirs]

        for label, point_dir in enumerate(point_dirs):
            for bag_dir in sorted(point_dir.iterdir()):
                if not bag_dir.is_dir():
                    continue
                for img_path in sorted(bag_dir.glob("*.jpg")):
                    self.samples.append((str(img_path), label))
        print(f"Loaded {len(self.samples)} images, {len(self.class_names)} classes: {self.class_names}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # dataset & split
    full_dataset = PointImageDataset(args.data_dir, transform=None)
    total = len(full_dataset)
    val_size = int(total * args.val_ratio)
    train_size = total - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(args.seed))

    # 给 subset 设置不同的 transform
    class TransformSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    train_loader = DataLoader(TransformSubset(train_subset, train_transform),
                              batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(TransformSubset(val_subset, val_transform),
                            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {train_size}, Val: {val_size}")

    # model
    model = PointClassifier(num_classes=10, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # training loop
    best_val_acc = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        # train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)

        scheduler.step()

        # val
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - total_start
        eta = epoch_time * (args.epochs - epoch)
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {epoch_time:.1f}s | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = save_dir / "point_classifier_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_names": full_dataset.class_names,
            }, ckpt_path)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    # save final
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "class_names": full_dataset.class_names,
    }, save_dir / "point_classifier_final.pth")

    print(f"\nTraining done. Best val acc: {best_val_acc:.4f}")
    print(f"Weights saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train point classifier")
    parser.add_argument("--data_dir", type=str,
                        default="/workspace/Mult-skill ACT/data/26-03-01-pick-image")
    parser.add_argument("--save_dir", type=str,
                        default="/workspace/Mult-skill ACT/models/point_classifier")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
