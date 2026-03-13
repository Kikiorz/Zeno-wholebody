"""
Bag to LeRobot V3.0 Dataset Converter
======================================
Dual-arm (7D+7D) + 4 cameras setup.

Topics:
  Cameras:
    /fisheye_left/image_raw/compressed
    /fisheye_right/image_raw/compressed
    /wide_mid/image_raw/compressed
    /wide_top/image_raw/compressed
  State:
    /robot/arm_left/joint_states_single   (7D)
    /robot/arm_right/joint_states_single  (7D)
  Action:
    /teleop/arm_left/joint_states_single  (7D)
    /teleop/arm_right/joint_states_single (7D)

Usage:
    conda activate mult-act
    python scripts/bag2lerobot_v30.py \
        --data-dir /workspace/Mult-skill\ ACT/data/YOUR_BAG_FOLDER \
        --repo-name your-dataset-name \
        --task "your task description"
"""

from pathlib import Path
import argparse
import numpy as np
import cv2
import time

from rosbags.highlevel import AnyReader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ============================================================
# DEFAULT CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"
DEFAULT_REPO_NAME = "mult-skill-act-v30"
DEFAULT_TASK = "dual arm manipulation task"

# ============================================================
# ROS TOPICS
# ============================================================
CAM_FISHEYE_LEFT = "/fisheye_left/image_raw/compressed"
CAM_FISHEYE_RIGHT = "/fisheye_right/image_raw/compressed"
CAM_WIDE_MID = "/wide_mid/image_raw/compressed"
CAM_WIDE_TOP = "/wide_top/image_raw/compressed"
ALL_CAM_TOPICS = [CAM_FISHEYE_LEFT, CAM_FISHEYE_RIGHT, CAM_WIDE_MID, CAM_WIDE_TOP]

STATE_LEFT = "/robot/arm_left/joint_states_single"
STATE_RIGHT = "/robot/arm_right/joint_states_single"
ACTION_LEFT = "/teleop/arm_left/joint_states_single"
ACTION_RIGHT = "/teleop/arm_right/joint_states_single"

ALL_TOPICS = set(ALL_CAM_TOPICS + [
    STATE_LEFT, STATE_RIGHT, ACTION_LEFT, ACTION_RIGHT,
])

# ============================================================
# SETTINGS
# ============================================================
FPS = 30
IMG_SIZE = (224, 224)
ARM_DOF = 7


def decode_compressed_image(msg):
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return img_rgb


def nearest_idx(times, t):
    idx = np.searchsorted(times, t)
    if idx == 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    return idx if abs(times[idx] - t) < abs(t - times[idx - 1]) else idx - 1


def extract_joint_positions(msg, dof=ARM_DOF):
    positions = list(msg.position)
    if len(positions) >= dof:
        return positions[:dof]
    return positions + [0.0] * (dof - len(positions))


def collect_bag_files(data_dir):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_path}")
        return []
    bag_files = sorted(data_path.glob("*.bag"))
    if not bag_files:
        bag_files = sorted(data_path.rglob("*.bag"))
    if bag_files:
        print(f"Found {len(bag_files)} bag file(s)")
    else:
        print(f"No .bag files found in {data_path}")
    return bag_files


def process_single_bag(bag_path, bag_idx, total_bags, task_label="dual arm manipulation task"):
    print(f"\n[Bag {bag_idx}/{total_bags}] Processing: {bag_path.name}")
    bag_start = time.time()

    try:
        with AnyReader([bag_path]) as reader:
            topic_to_msgs = {topic: [] for topic in ALL_TOPICS}
            connections = [c for c in reader.connections if c.topic in ALL_TOPICS]
            if not connections:
                print(f"  No relevant topics found, skipping.")
                return None

            for conn, t, raw in reader.messages(connections=connections):
                if conn.topic in topic_to_msgs:
                    msg = reader.deserialize(raw, conn.msgtype)
                    topic_to_msgs[conn.topic].append((t, msg))

            cam_fl = topic_to_msgs[CAM_FISHEYE_LEFT]
            cam_fr = topic_to_msgs[CAM_FISHEYE_RIGHT]
            cam_wm = topic_to_msgs[CAM_WIDE_MID]
            cam_wt = topic_to_msgs[CAM_WIDE_TOP]
            st_l = topic_to_msgs[STATE_LEFT]
            st_r = topic_to_msgs[STATE_RIGHT]
            ac_l = topic_to_msgs[ACTION_LEFT]
            ac_r = topic_to_msgs[ACTION_RIGHT]

            # Validate
            required = {"cam_fl": cam_fl, "cam_fr": cam_fr, "cam_wm": cam_wm,
                        "cam_wt": cam_wt, "st_l": st_l, "st_r": st_r,
                        "ac_l": ac_l, "ac_r": ac_r}
            for name, msgs in required.items():
                if not msgs:
                    print(f"  Missing {name}, skipping.")
                    return None

            # Use fisheye_left as reference timeline (~15Hz)
            ref_times = np.array([t for t, _ in cam_fl], dtype=np.int64)
            cam_fr_t = np.array([t for t, _ in cam_fr], dtype=np.int64)
            cam_wm_t = np.array([t for t, _ in cam_wm], dtype=np.int64)
            cam_wt_t = np.array([t for t, _ in cam_wt], dtype=np.int64)
            st_l_t = np.array([t for t, _ in st_l], dtype=np.int64)
            st_r_t = np.array([t for t, _ in st_r], dtype=np.int64)
            ac_l_t = np.array([t for t, _ in ac_l], dtype=np.int64)
            ac_r_t = np.array([t for t, _ in ac_r], dtype=np.int64)

            # Common time range
            t_start = max(ref_times[0], cam_fr_t[0], cam_wm_t[0], cam_wt_t[0],
                          st_l_t[0], st_r_t[0], ac_l_t[0], ac_r_t[0])
            t_end = min(ref_times[-1], cam_fr_t[-1], cam_wm_t[-1], cam_wt_t[-1],
                        st_l_t[-1], st_r_t[-1], ac_l_t[-1], ac_r_t[-1])

            # Resample at FPS within common range
            duration_s = (t_end - t_start) / 1e9
            n_frames = int(duration_s * FPS)
            if n_frames < 2:
                print(f"  Too short ({duration_s:.1f}s), skipping.")
                return None

            sample_times = np.linspace(t_start, t_end, n_frames, dtype=np.int64)
            print(f"  Duration: {duration_s:.1f}s, frames: {n_frames}")

            frames = []
            for i, t in enumerate(sample_times):
                # Decode 4 camera images
                img_fl = decode_compressed_image(cam_fl[nearest_idx(ref_times, t)][1])
                img_fr = decode_compressed_image(cam_fr[nearest_idx(cam_fr_t, t)][1])
                img_wm = decode_compressed_image(cam_wm[nearest_idx(cam_wm_t, t)][1])
                img_wt = decode_compressed_image(cam_wt[nearest_idx(cam_wt_t, t)][1])

                if any(x is None for x in [img_fl, img_fr, img_wm, img_wt]):
                    continue

                # State: left 7D + right 7D = 14D
                sl = extract_joint_positions(st_l[nearest_idx(st_l_t, t)][1])
                sr = extract_joint_positions(st_r[nearest_idx(st_r_t, t)][1])
                state = sl + sr

                # Action: left 7D + right 7D = 14D
                al = extract_joint_positions(ac_l[nearest_idx(ac_l_t, t)][1])
                ar = extract_joint_positions(ac_r[nearest_idx(ac_r_t, t)][1])
                action = al + ar

                frames.append({
                    "observation.images.fisheye_left": img_fl,
                    "observation.images.fisheye_right": img_fr,
                    "observation.images.wide_mid": img_wm,
                    "observation.images.wide_top": img_wt,
                    "observation.state": np.array(state, dtype=np.float32),
                    "action": np.array(action, dtype=np.float32),
                    "task": task_label,
                })

            elapsed = time.time() - bag_start
            print(f"  Done: {len(frames)} frames in {elapsed:.1f}s")
            return frames if frames else None

    except Exception as e:
        print(f"  Error processing {bag_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert ROS bags to LeRobot V3.0 dataset (dual-arm 7D+7D, 4 cameras)")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory containing .bag files")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output root directory")
    parser.add_argument("--repo-name", type=str, default=DEFAULT_REPO_NAME,
                        help="Dataset repo name")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="Task description label")
    parser.add_argument("--fps", type=int, default=FPS,
                        help="Target frame rate (default: 30)")
    parser.add_argument("--img-size", type=int, default=480,
                        help="Image resize dimension (default: 480)")
    args = parser.parse_args()

    fps = args.fps
    img_size = (args.img_size, args.img_size)

    output_path = Path(args.output_dir) / args.repo_name

    # Define features: 14D state, 14D action, 4 cameras
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (ARM_DOF * 2,),
            "names": [
                "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3",
                "left_joint_4", "left_joint_5", "left_joint_6",
                "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3",
                "right_joint_4", "right_joint_5", "right_joint_6",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (ARM_DOF * 2,),
            "names": [
                "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3",
                "left_joint_4", "left_joint_5", "left_joint_6",
                "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3",
                "right_joint_4", "right_joint_5", "right_joint_6",
            ],
        },
        "observation.images.fisheye_left": {
            "dtype": "video",
            "shape": (img_size[1], img_size[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.fisheye_right": {
            "dtype": "video",
            "shape": (img_size[1], img_size[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wide_mid": {
            "dtype": "video",
            "shape": (img_size[1], img_size[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wide_top": {
            "dtype": "video",
            "shape": (img_size[1], img_size[0], 3),
            "names": ["height", "width", "channels"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_name,
        root=output_path,
        robot_type="dual_arm",
        fps=fps,
        features=features,
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=4,
    )

    total_start = time.time()
    print(f"\n{'=' * 60}")
    print(f"Converting bags to LeRobot V3.0")
    print(f"  Task:    {args.task}")
    print(f"  Input:   {args.data_dir}")
    print(f"  Output:  {output_path}")
    print(f"  FPS:     {fps}")
    print(f"  ImgSize: {img_size}")
    print(f"  State:   {ARM_DOF * 2}D (left 7D + right 7D)")
    print(f"  Action:  {ARM_DOF * 2}D (left 7D + right 7D)")
    print(f"  Cameras: fisheye_left, fisheye_right, wide_mid, wide_top")
    print(f"{'=' * 60}")

    bag_files = collect_bag_files(args.data_dir)
    total_bags = len(bag_files)

    if total_bags == 0:
        print("No bag files to process.")
        return

    successful = 0
    for bag_idx, bag_path in enumerate(bag_files, 1):
        result = process_single_bag(bag_path, bag_idx, total_bags, task_label=args.task)
        if result is not None:
            for frame in result:
                dataset.add_frame(frame)
            dataset.save_episode()
            successful += 1
            print(f"  [Bag {bag_idx}/{total_bags}] Saved episode.")
        del result

    # CRITICAL: finalize flushes metadata_buffer (default buffers 10 episodes)
    # Without this, episodes not aligned to buffer size will be lost!
    dataset.finalize()

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Conversion complete!")
    print(f"  Episodes: {successful}/{total_bags}")
    print(f"  Time: {total_elapsed:.1f}s")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
