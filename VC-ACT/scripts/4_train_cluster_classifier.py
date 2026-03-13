"""
VC-ACT 阶段 4: 训练聚类分类器 (部署 Router)

训练一个 4-cam ResNet-18 分类器，输入 4 个相机的首帧，输出簇 ID。
部署时: 实时 4-cam 图像 → 此分类器 → 簇 ID → 对应簇的 ACT → 动作

输入:
  - results/cluster_labels_k10.npy 或 cluster_labels_auto.npy
  - results/episode_meta.json
  - bag_dir/ (原始 bag 文件)

输出:
  - models/exp_k10/cluster_classifier/best.pth
  - models/exp_auto/cluster_classifier/best.pth

用法:
  /venv/mult-act/bin/python3 4_train_cluster_classifier.py --mode k10
  /venv/mult-act/bin/python3 4_train_cluster_classifier.py --mode auto
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image

from rosbags.highlevel import AnyReader

CAM_TOPICS = [
    "/fisheye_left/image_raw/compressed",
    "/fisheye_right/image_raw/compressed",
    "/wide_mid/image_raw/compressed",
    "/wide_top/image_raw/compressed",
]
IMG_SIZE = (256, 256)


class FourCamClassifier(nn.Module):
    """4-camera ResNet-18 classifier.

    Each camera image goes through a shared ResNet-18 backbone (up to avgpool).
    4 x 512 features are concatenated and classified by a 2-layer MLP head.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 512, 1, 1)
        self.head = nn.Sequential(
            nn.Linear(512 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, imgs: list[torch.Tensor]) -> torch.Tensor:
        """imgs: list of 4 tensors, each (B, 3, 224, 224)."""
        feats = []
        for img in imgs:
            f = self.backbone(img).squeeze(-1).squeeze(-1)  # (B, 512)
            feats.append(f)
        x = torch.cat(feats, dim=1)  # (B, 2048)
        return self.head(x)


def decode_compressed_image(msg) -> np.ndarray | None:
    """Decode a ROS CompressedImage to RGB (H, W, 3) uint8."""
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)


class BagFirstFrameDataset(Dataset):
    """Dataset: first frame of each bag from 4 cameras, with cluster label."""

    def __init__(self, bag_dir: Path, episode_meta: list, cluster_labels: np.ndarray,
                 transform=None):
        self.bag_dir = bag_dir
        self.meta = episode_meta
        self.labels = cluster_labels
        self.transform = transform

        # Pre-extract all first frames to memory (only 100 bags, 4 images each, ~100MB)
        print("Pre-loading first frames from bags...")
        self.frames = []  # list of {topic: RGB_array}
        for i, m in enumerate(self.meta):
            bag_path = self.bag_dir / m["bag_name"]
            cam_imgs = self._extract_first_frame(bag_path)
            self.frames.append(cam_imgs)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(self.meta)}]")
        print(f"Loaded {len(self.frames)} episodes")

    def _extract_first_frame(self, bag_path: Path) -> dict[str, np.ndarray]:
        """Extract first frame from each camera in a bag."""
        result = {}
        try:
            with AnyReader([bag_path]) as reader:
                cam_conns = [c for c in reader.connections if c.topic in CAM_TOPICS]
                for conn, _t, raw in reader.messages(connections=cam_conns):
                    if conn.topic not in result:
                        msg = reader.deserialize(raw, conn.msgtype)
                        img = decode_compressed_image(msg)
                        if img is not None:
                            result[conn.topic] = img
                    if len(result) == 4:
                        break
        except Exception as e:
            print(f"  Error reading {bag_path.name}: {e}")
        return result

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        cam_imgs = self.frames[idx]
        label = int(self.labels[idx])

        imgs = []
        for topic in CAM_TOPICS:
            if topic in cam_imgs:
                img_rgb = cam_imgs[topic]
            else:
                img_rgb = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)

            if self.transform:
                img_pil = Image.fromarray(img_rgb)
                img_tensor = self.transform(img_pil)
            else:
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            imgs.append(img_tensor)

        return imgs, label


def collate_fn(batch):
    """Custom collate: list of (imgs_list, label) -> (4 batched tensors, labels)."""
    imgs_lists, labels = zip(*batch)
    batched_imgs = []
    for cam_idx in range(4):
        batched_imgs.append(torch.stack([imgs[cam_idx] for imgs in imgs_lists]))
    labels = torch.tensor(labels, dtype=torch.long)
    return batched_imgs, labels


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vc_root = Path(args.vc_root)
    results_dir = vc_root / "results"
    bag_dir = Path(args.bag_dir)

    # Load labels
    if args.mode == "k10":
        labels = np.load(results_dir / "cluster_labels_k10.npy")
        save_dir = vc_root / "models" / "exp_k10" / "cluster_classifier"
    else:
        labels = np.load(results_dir / "cluster_labels_auto.npy")
        save_dir = vc_root / "models" / "exp_auto" / "cluster_classifier"

    with open(results_dir / "episode_meta.json") as f:
        meta = json.load(f)

    num_classes = int(labels.max()) + 1
    print(f"Mode: {args.mode}, num_classes={num_classes}, total_bags={len(labels)}")
    print(f"Label distribution: {np.bincount(labels).tolist()}")

    # Transforms
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

    # Load dataset (pre-loads all first frames)
    full_dataset = BagFirstFrameDataset(bag_dir, meta, labels, transform=train_transform)

    # Split
    total = len(full_dataset)
    val_size = max(1, int(total * args.val_ratio))
    train_size = total - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Override transform for val subset
    class TransformOverride(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            # Get raw data from underlying dataset
            real_idx = self.subset.indices[idx]
            ds = self.subset.dataset
            cam_imgs = ds.frames[real_idx]
            label = int(ds.labels[real_idx])

            imgs = []
            for topic in CAM_TOPICS:
                if topic in cam_imgs:
                    img_rgb = cam_imgs[topic]
                else:
                    img_rgb = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
                img_pil = Image.fromarray(img_rgb)
                img_tensor = self.transform(img_pil)
                imgs.append(img_tensor)
            return imgs, label

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        TransformOverride(val_subset, val_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn
    )

    print(f"Train: {train_size}, Val: {val_size}")

    # Model
    model = FourCamClassifier(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels_batch in train_loader:
            imgs = [img.to(device) for img in imgs]
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels_batch.size(0)
            train_correct += (outputs.argmax(1) == labels_batch).sum().item()
            train_total += labels_batch.size(0)

        scheduler.step()

        # Val
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels_batch in val_loader:
                imgs = [img.to(device) for img in imgs]
                labels_batch = labels_batch.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item() * labels_batch.size(0)
                val_correct += (outputs.argmax(1) == labels_batch).sum().item()
                val_total += labels_batch.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {train_loss/max(train_total,1):.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss/max(val_total,1):.4f} Acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
                "mode": args.mode,
            }, save_dir / "best.pth")
            print(f"  -> Saved best (val_acc={val_acc:.4f})")

    # Save final
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "num_classes": num_classes,
        "mode": args.mode,
    }, save_dir / "final.pth")

    print(f"\nDone. Best val acc: {best_val_acc:.4f}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VC-ACT: Train 4-cam cluster classifier")
    parser.add_argument("--mode", type=str, choices=["k10", "auto"], required=True)
    parser.add_argument("--vc_root", type=str,
                        default="/workspace/Mult-skill ACT/VC-ACT")
    parser.add_argument("--bag_dir", type=str,
                        default="/workspace/Mult-skill ACT/data/26-03-04-pick_hanger")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
