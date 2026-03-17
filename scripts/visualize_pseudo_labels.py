#!/usr/bin/env python3
"""
Visualize pseudo-labels as annotated MP4 videos for human verification.

For each episode, renders 3 camera views side-by-side with:
- Color-coded borders (blue=stageA, orange=stageB, red=switch zone)
- Progress bar with τ* marker
- Text overlay with current label info and error values
- Mini error curve with current-position marker

Usage:
    python visualize_pseudo_labels.py --episodes 0 --speedup 4
    python visualize_pseudo_labels.py --episodes 0,1,5 --speedup 2
    python visualize_pseudo_labels.py  # all episodes, 2x speed
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lerobot" / "src"))

from lerobot.datasets.video_utils import decode_video_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT / "data" / "Exp_stageAB"
STAGE3_DIR = DATA_ROOT / "stage3_lerobot"
LABEL_ROOT = DATA_ROOT / "pseudo_labels"

CAMERAS = ["realsense_top", "realsense_left", "realsense_right"]
FPS = 30

# Colors (BGR for OpenCV)
COLOR_STAGE_A = (255, 150, 50)    # blue-ish
COLOR_STAGE_B = (50, 150, 255)    # orange-ish
COLOR_SWITCH = (0, 0, 255)        # red
COLOR_BG = (30, 30, 30)           # dark background
COLOR_TEXT = (220, 220, 220)       # light text
COLOR_PROGRESS_BG = (60, 60, 60)
COLOR_PROGRESS_A = (200, 130, 50)
COLOR_PROGRESS_B = (50, 130, 200)
COLOR_TAU_MARKER = (0, 0, 255)

# Layout constants
CAM_SIZE = 224
BORDER_W = 4
CAM_WITH_BORDER = CAM_SIZE + 2 * BORDER_W   # 232
CANVAS_W = 3 * CAM_WITH_BORDER + 4 * 8      # 3 cams + gaps
INFO_H = 70
CHART_H = 140
CANVAS_H = CAM_WITH_BORDER + INFO_H + CHART_H
MARGIN = 8


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_episode_metadata():
    ep_pq = STAGE3_DIR / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(str(ep_pq))
    return table.to_pandas()


def load_episode_timestamps(ep_idx, ep_meta):
    data_pq = STAGE3_DIR / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(str(data_pq), columns=["timestamp", "episode_index"])
    df = table.to_pandas()
    mask = df["episode_index"] == ep_idx
    return df.loc[mask, "timestamp"].values.astype(np.float64)


def decode_episode_frames(ep_idx, ep_meta, timestamps):
    """Decode video frames for all 3 cameras. Returns dict: cam -> (T, 3, H, W) float32 [0,1]."""
    row = ep_meta.loc[ep_meta["episode_index"] == ep_idx].iloc[0]
    frames = {}
    for cam in CAMERAS:
        chunk_idx = int(row[f"videos/observation.images.{cam}/chunk_index"])
        file_idx = int(row[f"videos/observation.images.{cam}/file_index"])
        from_ts = float(row[f"videos/observation.images.{cam}/from_timestamp"])
        video_path = (STAGE3_DIR / "videos" / f"observation.images.{cam}"
                      / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4")
        video_timestamps = (timestamps + from_ts).tolist()
        cam_frames = decode_video_frames(
            video_path=video_path,
            timestamps=video_timestamps,
            tolerance_s=1.0 / FPS + 0.01,
            backend="pyav",
        )
        frames[cam] = cam_frames
    return frames


def load_pseudo_labels(ep_idx):
    pkl_path = LABEL_ROOT / "episodes" / f"episode_{ep_idx:03d}.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ─── Rendering Helpers ───────────────────────────────────────────────────────

def prerender_error_chart(e_A, e_B, tau_star, T, width, height):
    """Pre-render the error curve as a static background image (numpy BGR)."""
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    ax.plot(e_A, color="#3296fa", linewidth=0.6, alpha=0.85, label="e_A (stageA)")
    ax.plot(e_B, color="#fa9632", linewidth=0.6, alpha=0.85, label="e_B (stageB)")
    ax.axvline(tau_star, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(0, T - 1)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_color("#555555")
    ax.legend(loc="upper right", fontsize=7, facecolor="#2a2a2a", edgecolor="#555555",
              labelcolor="#cccccc")
    ax.set_ylabel("L2 error", fontsize=7, color="#aaaaaa")

    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    # RGBA -> BGR, resize to exact target
    chart_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    chart_bgr = cv2.resize(chart_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return chart_bgr


def add_border(frame_bgr, color, border_w=BORDER_W):
    """Add a colored border around a frame."""
    bordered = cv2.copyMakeBorder(
        frame_bgr, border_w, border_w, border_w, border_w,
        cv2.BORDER_CONSTANT, value=color,
    )
    return bordered


def draw_progress_bar(canvas, y, x_start, x_end, t, T, tau_star):
    """Draw progress bar with τ* marker."""
    bar_h = 12
    # Background
    cv2.rectangle(canvas, (x_start, y), (x_end, y + bar_h), COLOR_PROGRESS_BG, -1)

    # Filled portion
    frac = t / max(T - 1, 1)
    fill_x = int(x_start + frac * (x_end - x_start))
    bar_color = COLOR_PROGRESS_A if t <= tau_star else COLOR_PROGRESS_B
    cv2.rectangle(canvas, (x_start, y), (fill_x, y + bar_h), bar_color, -1)

    # τ* marker
    tau_x = int(x_start + (tau_star / max(T - 1, 1)) * (x_end - x_start))
    cv2.line(canvas, (tau_x, y - 2), (tau_x, y + bar_h + 2), COLOR_TAU_MARKER, 2)

    # Small triangle above τ*
    pts = np.array([[tau_x - 4, y - 5], [tau_x + 4, y - 5], [tau_x, y - 1]], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], COLOR_TAU_MARKER)


def draw_current_position_line(chart_bg, t, T, chart_x, chart_y, canvas):
    """Overlay the chart background + a vertical line for current position onto canvas."""
    h, w = chart_bg.shape[:2]
    # Clip to canvas bounds
    avail_h = canvas.shape[0] - chart_y
    avail_w = canvas.shape[1] - chart_x
    h = min(h, avail_h)
    w = min(w, avail_w)
    canvas[chart_y:chart_y + h, chart_x:chart_x + w] = chart_bg[:h, :w]

    # Draw vertical line at current t
    # The matplotlib plot x-axis spans from ~left_margin to ~right_margin within the chart image
    # We approximate the plot area as roughly 10%-95% of the chart width
    plot_left = int(w * 0.09)
    plot_right = int(w * 0.97)
    frac = t / max(T - 1, 1)
    line_x = chart_x + plot_left + int(frac * (plot_right - plot_left))
    cv2.line(canvas, (line_x, chart_y + 5), (line_x, chart_y + h - 10),
             (255, 255, 255), 1, cv2.LINE_AA)


# ─── Main Rendering Loop ────────────────────────────────────────────────────

def render_episode_video(ep_idx, ep_meta, args):
    """Render a visualization video for one episode."""
    logger.info(f"=== Episode {ep_idx} ===")

    # Load label data
    labels = load_pseudo_labels(ep_idx)
    T = labels["T"]
    tau_star = labels["tau_star"]
    e_A = labels["e_A"]
    e_B = labels["e_B"]
    hard_labels = labels["hard_labels"]
    soft_labels = labels["soft_labels"]

    logger.info(f"  T={T}, τ*={tau_star} ({tau_star/T*100:.1f}%)")

    # Load video frames
    logger.info("  Decoding video frames...")
    timestamps = load_episode_timestamps(ep_idx, ep_meta)
    frames = decode_episode_frames(ep_idx, ep_meta, timestamps)
    logger.info("  Video decode done.")

    # Convert frames to numpy uint8 (H, W, 3) BGR for each camera
    # Input is (T, 3, H, W) float32 [0,1]
    cam_frames = {}
    for cam in CAMERAS:
        f = frames[cam].numpy()                     # (T, 3, 224, 224)
        f = (f * 255).clip(0, 255).astype(np.uint8)
        f = f.transpose(0, 2, 3, 1)                # (T, 224, 224, 3) RGB
        f = f[:, :, :, ::-1].copy()                 # RGB -> BGR
        cam_frames[cam] = f
    del frames

    # Pre-render error chart
    chart_w = CANVAS_W - 2 * MARGIN
    chart_img = prerender_error_chart(e_A, e_B, tau_star, T, chart_w, CHART_H)

    # Setup video writer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"episode_{ep_idx:03d}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (CANVAS_W, CANVAS_H))

    if not writer.isOpened():
        logger.error(f"  Failed to open video writer for {out_path}")
        return

    # Ambiguity window for red border marking
    amb_window = 30  # frames around tau_star to highlight

    # Render frames
    speedup = args.speedup
    frame_indices = list(range(0, T, speedup))
    logger.info(f"  Rendering {len(frame_indices)} frames (speedup={speedup}x)...")

    for count, t in enumerate(frame_indices):
        canvas = np.full((CANVAS_H, CANVAS_W, 3), COLOR_BG, dtype=np.uint8)

        # --- Top row: 3 camera views with colored borders ---
        is_near_switch = abs(t - tau_star) <= amb_window
        if is_near_switch:
            border_color = COLOR_SWITCH
        elif hard_labels[t] == 0:
            border_color = COLOR_STAGE_A
        else:
            border_color = COLOR_STAGE_B

        x_offset = MARGIN
        for cam in CAMERAS:
            frame_bgr = cam_frames[cam][t]
            bordered = add_border(frame_bgr, border_color)
            canvas[MARGIN:MARGIN + CAM_WITH_BORDER, x_offset:x_offset + CAM_WITH_BORDER] = bordered
            x_offset += CAM_WITH_BORDER + MARGIN

        # --- Info panel ---
        info_y = MARGIN + CAM_WITH_BORDER + 4

        # Progress bar
        draw_progress_bar(canvas, info_y, MARGIN, CANVAS_W - MARGIN, t, T, tau_star)

        # Text info
        text_y = info_y + 20
        expert_str = "stageA" if hard_labels[t] == 0 else "stageB"
        soft_b = soft_labels[t]

        line1 = f"t={t}/{T}  Expert: {expert_str}  Soft P(B)={soft_b:.3f}"
        cv2.putText(canvas, line1, (MARGIN, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

        line2 = f"e_A={e_A[t]:.3f}  e_B={e_B[t]:.3f}  tau*={tau_star}"
        cv2.putText(canvas, line2, (MARGIN, text_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

        # Stage indicator colored dot
        dot_x = CANVAS_W - MARGIN - 60
        cv2.circle(canvas, (dot_x, text_y - 4), 6, border_color, -1)
        stage_label = "A" if hard_labels[t] == 0 else "B"
        cv2.putText(canvas, stage_label, (dot_x + 12, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 1, cv2.LINE_AA)

        # --- Mini error chart ---
        chart_y = MARGIN + CAM_WITH_BORDER + INFO_H
        draw_current_position_line(chart_img, t, T, MARGIN, chart_y, canvas)

        writer.write(canvas)

        if count % 200 == 0:
            logger.info(f"    frame {count}/{len(frame_indices)}")

    writer.release()
    logger.info(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize pseudo-labels as annotated videos")
    parser.add_argument("--episodes", type=str, default=None,
                        help="Comma-separated episode indices or range (e.g. '0' or '0,1,5' or '0-19'). Default: all.")
    parser.add_argument("--speedup", type=int, default=2,
                        help="Playback speed multiplier (skip frames). Default: 2")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Default: pseudo_labels/videos/")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video FPS. Default: 30")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(LABEL_ROOT / "videos")

    # Parse episode list
    if args.episodes is None:
        episode_list = list(range(20))
    elif "-" in args.episodes and "," not in args.episodes:
        lo, hi = args.episodes.split("-")
        episode_list = list(range(int(lo), int(hi) + 1))
    else:
        episode_list = [int(x) for x in args.episodes.split(",")]

    logger.info(f"Episodes: {episode_list}")
    logger.info(f"Speedup: {args.speedup}x, FPS: {args.fps}")
    logger.info(f"Output: {args.output_dir}")

    ep_meta = load_episode_metadata()

    for ep_idx in episode_list:
        render_episode_video(ep_idx, ep_meta, args)

    logger.info("All done.")


if __name__ == "__main__":
    main()
