#!/usr/bin/env python3
"""Inference router on all 20 episodes, visualize P(B) predictions vs pseudo-labels."""

import pickle
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from router.model import TemporalProgressRouter
from router.dataset import RouterDataset, STAGE3_DIR, LABEL_DIR, CAMERAS, FPS, IMAGE_MEAN, IMAGE_STD, CHUNK_SIZE

import pyarrow.parquet as pq
from lerobot.datasets.video_utils import decode_video_frames


def load_router(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TemporalProgressRouter(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded router from {ckpt_path} (step {ckpt['step']})")
    return model


@torch.no_grad()
def infer_episode(model, ep_data, ep_meta_row, state_mean, state_std, device="cuda", batch_size=32):
    """Run router on every valid timestep of an episode. Returns (T-99,100) P(B) array."""
    states = ep_data["states"]
    timestamps = ep_data["timestamps"]
    T = len(states)
    n_valid = T - CHUNK_SIZE + 1

    all_logits = []

    for start in range(0, n_valid, batch_size):
        end = min(start + batch_size, n_valid)
        B = end - start

        # Normalize states
        batch_states = torch.from_numpy(
            (states[start:end] - state_mean) / state_std
        ).float().to(device)

        # Decode images for this batch
        batch_images = [[] for _ in range(len(CAMERAS))]
        for b_idx in range(B):
            t = start + b_idx
            ts = timestamps[t]
            for c_idx, cam in enumerate(CAMERAS):
                chunk_idx = int(ep_meta_row[f"videos/observation.images.{cam}/chunk_index"])
                file_idx = int(ep_meta_row[f"videos/observation.images.{cam}/file_index"])
                from_ts = float(ep_meta_row[f"videos/observation.images.{cam}/from_timestamp"])
                video_path = (
                    STAGE3_DIR / "videos" / f"observation.images.{cam}"
                    / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
                )
                frame = decode_video_frames(
                    video_path=video_path,
                    timestamps=[float(ts + from_ts)],
                    tolerance_s=1.0 / FPS + 0.01,
                    backend="pyav",
                )  # (1, 3, H, W)
                frame = (frame[0] - IMAGE_MEAN) / IMAGE_STD
                batch_images[c_idx].append(frame)

        images = [torch.stack(batch_images[c]).to(device) for c in range(len(CAMERAS))]

        logits = model(images, batch_states)  # (B, 100)
        all_logits.append(logits.cpu())

        if (start // batch_size) % 10 == 0:
            print(f"    {end}/{n_valid}", end="\r", flush=True)

    all_logits = torch.cat(all_logits, dim=0)  # (n_valid, 100)
    probs = torch.sigmoid(all_logits).numpy()
    print(f"    Done: {n_valid} timesteps")
    return probs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = PROJECT / "model" / "router" / "checkpoints" / "step_50000.pt"
    out_dir = PROJECT / "model" / "router" / "eval_step50k"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_router(ckpt_path, device)

    # Load dataset metadata
    ep_parquet = STAGE3_DIR / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_meta = pq.read_table(str(ep_parquet)).to_pandas()
    data_parquet = STAGE3_DIR / "data" / "chunk-000" / "file-000.parquet"
    all_data = pq.read_table(str(data_parquet))

    stats_path = STAGE3_DIR / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)
    state_mean = np.array(stats["observation.state"]["mean"], dtype=np.float32)
    state_std = np.clip(np.array(stats["observation.state"]["std"], dtype=np.float32), 1e-6, None)

    episode_indices = sorted(ep_meta["episode_index"].unique())

    for ep_idx in episode_indices:
        print(f"\nEpisode {ep_idx}:")
        row = ep_meta.loc[ep_meta["episode_index"] == ep_idx].iloc[0]
        from_idx = int(row["dataset_from_index"])
        to_idx = int(row["dataset_to_index"])

        states = np.array(all_data["observation.state"][from_idx:to_idx].to_pylist(), dtype=np.float32)
        timestamps = np.array(all_data["timestamp"][from_idx:to_idx].to_pylist(), dtype=np.float64)

        with open(LABEL_DIR / f"episode_{ep_idx:03d}.pkl", "rb") as f:
            labels = pickle.load(f)

        ep_data = {"states": states, "timestamps": timestamps}
        probs = infer_episode(model, ep_data, row, state_mean, state_std, device, batch_size=16)

        # probs shape: (T-99, 100)
        T = len(states)
        tau = labels["tau_star"]
        soft_labels = labels["soft_labels"]

        # Extract p[0] for each timestep (the "current step" prediction)
        p0_router = probs[:, 0]  # (T-99,)

        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

        # Panel 1: p[0] comparison
        ax = axes[0]
        ax.plot(soft_labels, alpha=0.5, linewidth=0.5, label="soft_label (pseudo)", color="gray")
        ax.plot(range(len(p0_router)), p0_router, linewidth=1.0, label="router p[0]", color="blue")
        ax.axvline(tau, color="red", linestyle="--", linewidth=0.8, label=f"τ*={tau}")
        ax.set_ylabel("P(expert_B)")
        ax.set_title(f"Episode {ep_idx} — Router vs Pseudo-labels (step 50K)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: chunk heatmap from router
        ax = axes[1]
        # Show every 50th timestep's full chunk prediction
        step_indices = list(range(0, len(probs), 50))
        chunk_matrix = probs[step_indices, :]  # (N, 100)
        im = ax.imshow(chunk_matrix.T, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1,
                       extent=[step_indices[0], step_indices[-1], 99, 0])
        ax.axvline(tau, color="red", linestyle="--", linewidth=0.8)
        ax.set_ylabel("chunk index (0=now, 99=future)")
        ax.set_title("Router chunk predictions (sampled every 50 steps)")
        plt.colorbar(im, ax=ax, label="P(B)")

        # Panel 3: crossing_idx over time
        ax = axes[2]
        crossing_indices = []
        for i in range(len(probs)):
            cross = np.where(probs[i] > 0.5)[0]
            crossing_indices.append(cross[0] if len(cross) > 0 else 100)
        ax.plot(crossing_indices, linewidth=0.8, color="green")
        ax.axvline(tau, color="red", linestyle="--", linewidth=0.8, label=f"τ*={tau}")
        ax.set_ylabel("crossing_idx")
        ax.set_xlabel("timestep")
        ax.set_title("Crossing index (first chunk position where P(B)>0.5)")
        ax.set_ylim(-5, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(out_dir / f"episode_{ep_idx:03d}.png", dpi=120)
        plt.close(fig)
        print(f"  Saved plot → {out_dir / f'episode_{ep_idx:03d}.png'}")

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
