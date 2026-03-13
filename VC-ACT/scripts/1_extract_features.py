"""
VC-ACT 阶段 1: 从 rosbag 提取视觉特征

从指定文件夹的 .bag 文件中，为每条 bag 提取第 1 帧的 4 相机 ResNet-18 特征。
输出:
  - results/features_2048.npy   (N, 2048) float32
  - results/episode_meta.json   [{bag_index, bag_name}, ...]

用法:
  /venv/mult-act/bin/python3 1_extract_features.py
  /venv/mult-act/bin/python3 1_extract_features.py --bag_dir /path/to/bags
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from rosbags.highlevel import AnyReader

# ROS camera topics (same as bag2lerobot_v30.py)
CAM_TOPICS = [
    "/fisheye_left/image_raw/compressed",
    "/fisheye_right/image_raw/compressed",
    "/wide_mid/image_raw/compressed",
    "/wide_top/image_raw/compressed",
]

IMG_SIZE = (256, 256)

# ImageNet normalization
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),  # HWC uint8 -> CHW float [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_feature_extractor(device: torch.device) -> nn.Module:
    """ResNet-18 without the final FC layer. Output: (B, 512) after AvgPool."""
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    extractor = nn.Sequential(*list(resnet.children())[:-1])  # output: (B, 512, 1, 1)
    extractor.eval()
    return extractor.to(device)


def decode_compressed_image(msg) -> np.ndarray | None:
    """Decode a ROS CompressedImage message to RGB (H, W, 3) uint8."""
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return img_rgb


def extract_first_frame_from_bag(bag_path: Path) -> dict[str, np.ndarray] | None:
    """Extract the first frame from each of 4 cameras in a rosbag.

    Returns dict: {topic: RGB image (H,W,3)} or None on failure.
    """
    try:
        with AnyReader([bag_path]) as reader:
            # Find connections for camera topics
            cam_connections = [c for c in reader.connections if c.topic in CAM_TOPICS]
            if len(cam_connections) < 4:
                print(f"    Warning: only {len(cam_connections)}/4 camera topics found")
                return None

            # Collect first message per camera
            first_frames = {}
            for conn, _t, raw in reader.messages(connections=cam_connections):
                if conn.topic not in first_frames:
                    msg = reader.deserialize(raw, conn.msgtype)
                    img = decode_compressed_image(msg)
                    if img is not None:
                        first_frames[conn.topic] = img
                if len(first_frames) == 4:
                    break

            if len(first_frames) < 4:
                missing = set(CAM_TOPICS) - set(first_frames.keys())
                print(f"    Warning: missing cameras: {missing}")
                return None

            return first_frames

    except Exception as e:
        print(f"    Error reading bag: {e}")
        return None


def main(args):
    bag_dir = Path(args.bag_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Find all bag files
    bag_files = sorted(bag_dir.glob("*.bag"))
    n_bags = len(bag_files)
    print(f"Found {n_bags} bag files in {bag_dir}")
    if n_bags == 0:
        print("No .bag files found, exiting.")
        return

    # Build feature extractor
    extractor = build_feature_extractor(device)

    all_features = []
    episode_meta = []
    failed = []

    with torch.no_grad():
        for i, bag_path in enumerate(bag_files):
            frames = extract_first_frame_from_bag(bag_path)
            if frames is None:
                print(f"  [{i+1}/{n_bags}] SKIP {bag_path.name}")
                failed.append(bag_path.name)
                continue

            # Extract ResNet-18 features for each camera and concat
            cam_features = []
            for topic in CAM_TOPICS:
                img_rgb = frames[topic]
                img_tensor = TRANSFORM(img_rgb).unsqueeze(0).to(device)  # (1, 3, 256, 256)
                feat = extractor(img_tensor)  # (1, 512, 1, 1)
                feat = feat.squeeze()  # (512,)
                cam_features.append(feat)

            episode_feat = torch.cat(cam_features, dim=0).cpu().numpy()  # (2048,)
            all_features.append(episode_feat)
            episode_meta.append({
                "bag_index": i,
                "bag_name": bag_path.name,
            })

            if (i + 1) % 10 == 0 or i == n_bags - 1:
                print(f"  [{i+1}/{n_bags}] {bag_path.name} OK")

    features = np.stack(all_features, axis=0)  # (N, 2048)
    print(f"\nFeature matrix shape: {features.shape}")
    if failed:
        print(f"Failed bags ({len(failed)}): {failed}")

    # Save
    feat_path = results_dir / "features_2048.npy"
    meta_path = results_dir / "episode_meta.json"
    np.save(feat_path, features)
    with open(meta_path, "w") as f:
        json.dump(episode_meta, f, indent=2)

    print(f"Saved features to {feat_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VC-ACT: Extract 4-cam ResNet-18 features from rosbags")
    parser.add_argument("--bag_dir", type=str,
                        default="/workspace/Mult-skill ACT/data/26-03-04-pick_hanger")
    parser.add_argument("--results_dir", type=str,
                        default="/workspace/Mult-skill ACT/VC-ACT/results")
    args = parser.parse_args()
    main(args)
