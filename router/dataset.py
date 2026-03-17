"""RouterDataset — loads stage3 trajectories + pseudo-labels for router training."""

import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lerobot" / "src"))
from lerobot.datasets.video_utils import decode_video_frames

# ── Constants ────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT / "data" / "Exp_stageAB"
STAGE3_DIR = DATA_ROOT / "stage3_lerobot"
LABEL_DIR = DATA_ROOT / "pseudo_labels" / "episodes"

CAMERAS = ["realsense_top", "realsense_left", "realsense_right"]
CHUNK_SIZE = 100
FPS = 30

# ImageNet normalization
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)


class RouterDataset(Dataset):
    def __init__(
        self,
        stage3_dir: Path | str | None = None,
        label_dir: Path | str | None = None,
    ):
        stage3_dir = Path(stage3_dir) if stage3_dir else STAGE3_DIR
        label_dir = Path(label_dir) if label_dir else LABEL_DIR

        # ── Load episode metadata ────────────────────────────────────────
        ep_parquet = stage3_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        self.ep_meta = pq.read_table(str(ep_parquet)).to_pandas()

        # ── Load all data (states + timestamps) ─────────────────────────
        data_parquet = stage3_dir / "data" / "chunk-000" / "file-000.parquet"
        all_data = pq.read_table(str(data_parquet))

        # ── Load state normalization stats ───────────────────────────────
        stats_path = stage3_dir / "meta" / "stats.json"
        with open(stats_path) as f:
            stats = json.load(f)
        self.state_mean = np.array(stats["observation.state"]["mean"], dtype=np.float32)
        self.state_std = np.array(stats["observation.state"]["std"], dtype=np.float32)
        # Clamp std to avoid div-by-zero
        self.state_std = np.clip(self.state_std, 1e-6, None)

        # ── Per-episode data ─────────────────────────────────────────────
        self.episodes = []  # list of dicts with states, timestamps, soft_labels, ep_meta_row
        self.index = []     # flat list of (ep_list_idx, t)

        episode_indices = sorted(self.ep_meta["episode_index"].unique())
        for ep_idx in episode_indices:
            row = self.ep_meta.loc[self.ep_meta["episode_index"] == ep_idx].iloc[0]
            from_idx = int(row["dataset_from_index"])
            to_idx = int(row["dataset_to_index"])

            states = np.array(
                all_data["observation.state"][from_idx:to_idx].to_pylist(), dtype=np.float32
            )
            timestamps = np.array(
                all_data["timestamp"][from_idx:to_idx].to_pylist(), dtype=np.float64
            )

            # Load pseudo-labels
            label_path = label_dir / f"episode_{ep_idx:03d}.pkl"
            with open(label_path, "rb") as f:
                labels = pickle.load(f)
            soft_labels = labels["soft_labels"].astype(np.float32)

            T = len(states)
            assert len(soft_labels) == T, (
                f"Episode {ep_idx}: states len {T} != soft_labels len {len(soft_labels)}"
            )

            ep_list_idx = len(self.episodes)
            self.episodes.append({
                "states": states,
                "timestamps": timestamps,
                "soft_labels": soft_labels,
                "ep_idx": ep_idx,
                "ep_meta_row": row,
            })

            # Index: t ∈ [0, T - chunk_size]
            for t in range(T - CHUNK_SIZE + 1):
                self.index.append((ep_list_idx, t))

        print(f"[RouterDataset] {len(self.episodes)} episodes, {len(self.index)} samples")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_list_idx, t = self.index[idx]
        ep = self.episodes[ep_list_idx]

        # ── State (normalized) ───────────────────────────────────────────
        state = (ep["states"][t] - self.state_mean) / self.state_std  # (17,)

        # ── Labels ───────────────────────────────────────────────────────
        labels = ep["soft_labels"][t : t + CHUNK_SIZE]  # (100,)

        # ── Images (decode single frame per camera) ──────────────────────
        row = ep["ep_meta_row"]
        timestamp = ep["timestamps"][t]
        images = []
        for cam in CAMERAS:
            chunk_idx = int(row[f"videos/observation.images.{cam}/chunk_index"])
            file_idx = int(row[f"videos/observation.images.{cam}/file_index"])
            from_ts = float(row[f"videos/observation.images.{cam}/from_timestamp"])

            video_path = (
                STAGE3_DIR / "videos" / f"observation.images.{cam}"
                / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
            )

            video_ts = float(timestamp + from_ts)
            frame = decode_video_frames(
                video_path=video_path,
                timestamps=[video_ts],
                tolerance_s=1.0 / FPS + 0.01,
                backend="pyav",
            )  # (1, 3, H, W) float32 in [0, 1]

            # ImageNet normalize
            frame = (frame[0] - IMAGE_MEAN) / IMAGE_STD  # (3, 224, 224)
            images.append(frame)

        # images: list of 3 tensors each (3, 224, 224)
        # state: numpy (17,) -> tensor
        # labels: numpy (100,) -> tensor
        return images, torch.from_numpy(state), torch.from_numpy(labels)


def router_collate_fn(batch):
    """Collate into per-camera image tensors.

    Returns:
        images: list of 3 tensors each (B, 3, 224, 224)
        states: (B, 17)
        labels: (B, 100)
    """
    images_list, states, labels = zip(*batch)
    n_cams = len(images_list[0])
    images = [torch.stack([sample[c] for sample in images_list]) for c in range(n_cams)]
    states = torch.stack(states)
    labels = torch.stack(labels)
    return images, states, labels
