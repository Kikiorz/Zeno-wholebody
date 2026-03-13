"""Bag to LeRobot V3.0 Dataset Converter for Exp_stageAB
=======================================================
Dual-arm (7D+7D) + 3 cameras + odom.

Topics:
  Cameras:
    /realsense_top/color/image_raw/compressed
    /realsense_left/color/image_raw/compressed
    /realsense_right/color/image_raw/compressed
  State:
    /robot/arm_left/joint_states_single   (7D)
    /robot/arm_right/joint_states_single  (7D)
  Action:
    /teleop/arm_left/joint_states_single  (7D)
    /teleop/arm_right/joint_states_single (7D)
  Odom:
    /ranger_base_node/odom

Usage:
    conda activate base
    python convert_exp.py stage1
    python convert_exp.py stage2
    python convert_exp.py stage3
"""

import sys
from pathlib import Path
import shutil
import numpy as np
import cv2
import time

from rosbags.highlevel import AnyReader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ============================================================
# CONFIGURATION
# ============================================================
STAGE = sys.argv[1] if len(sys.argv) > 1 else "stage1"
DATA_ROOT = Path(f"/home/zeno-rp/2026CoRL/Data/Exp_stageAB/{STAGE}")
OUTPUT_DIR = Path("/home/zeno-rp/2026CoRL/Data/Exp_stageAB")

TASK_NAMES = {
    "stage1": "hanger",
    "stage2": "move",
    "stage3": "move_hanger",
}
TASK_NAME = TASK_NAMES.get(STAGE, STAGE)

# ROS topics
CAM_TOP = "/realsense_top/color/image_raw/compressed"
CAM_LEFT = "/realsense_left/color/image_raw/compressed"
CAM_RIGHT = "/realsense_right/color/image_raw/compressed"
STATE_LEFT = "/robot/arm_left/joint_states_single"
STATE_RIGHT = "/robot/arm_right/joint_states_single"
ACTION_LEFT = "/teleop/arm_left/joint_states_single"
ACTION_RIGHT = "/teleop/arm_right/joint_states_single"
ODOM = "/ranger_base_node/odom"

ALL_TOPICS = {
    CAM_TOP, CAM_LEFT, CAM_RIGHT,
    STATE_LEFT, STATE_RIGHT, ACTION_LEFT, ACTION_RIGHT, ODOM,
}

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


def process_single_bag(bag_path, bag_idx, total_bags, task_label):
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

            cam_top = topic_to_msgs[CAM_TOP]
            cam_left = topic_to_msgs[CAM_LEFT]
            cam_right = topic_to_msgs[CAM_RIGHT]
            st_l = topic_to_msgs[STATE_LEFT]
            st_r = topic_to_msgs[STATE_RIGHT]
            ac_l = topic_to_msgs[ACTION_LEFT]
            ac_r = topic_to_msgs[ACTION_RIGHT]
            odom_msgs = topic_to_msgs[ODOM]

            required = {
                "cam_top": cam_top, "cam_left": cam_left, "cam_right": cam_right,
                "st_l": st_l, "st_r": st_r, "ac_l": ac_l, "ac_r": ac_r,
            }
            for name, msgs in required.items():
                if not msgs:
                    print(f"  Missing {name}, skipping.")
                    return None

            cam_top_t = np.array([t for t, _ in cam_top], dtype=np.int64)
            cam_left_t = np.array([t for t, _ in cam_left], dtype=np.int64)
            cam_right_t = np.array([t for t, _ in cam_right], dtype=np.int64)
            st_l_t = np.array([t for t, _ in st_l], dtype=np.int64)
            st_r_t = np.array([t for t, _ in st_r], dtype=np.int64)
            ac_l_t = np.array([t for t, _ in ac_l], dtype=np.int64)
            ac_r_t = np.array([t for t, _ in ac_r], dtype=np.int64)
            odom_t = (
                np.array([t for t, _ in odom_msgs], dtype=np.int64)
                if odom_msgs else np.array([], dtype=np.int64)
            )

            t_start = max(cam_top_t[0], cam_left_t[0], cam_right_t[0],
                          st_l_t[0], st_r_t[0], ac_l_t[0], ac_r_t[0])
            t_end = min(cam_top_t[-1], cam_left_t[-1], cam_right_t[-1],
                        st_l_t[-1], st_r_t[-1], ac_l_t[-1], ac_r_t[-1])

            duration_s = (t_end - t_start) / 1e9
            n_frames = int(duration_s * FPS)
            if n_frames < 2:
                print(f"  Too short ({duration_s:.1f}s), skipping.")
                return None

            sample_times = np.linspace(t_start, t_end, n_frames, dtype=np.int64)
            print(f"  Duration: {duration_s:.1f}s, frames: {n_frames}")

            frames = []
            for i, t in enumerate(sample_times):
                img_top = decode_compressed_image(cam_top[nearest_idx(cam_top_t, t)][1])
                img_left = decode_compressed_image(cam_left[nearest_idx(cam_left_t, t)][1])
                img_right = decode_compressed_image(cam_right[nearest_idx(cam_right_t, t)][1])

                if any(x is None for x in [img_top, img_left, img_right]):
                    continue

                sl = extract_joint_positions(st_l[nearest_idx(st_l_t, t)][1])
                sr = extract_joint_positions(st_r[nearest_idx(st_r_t, t)][1])
                state = sl + sr

                al = extract_joint_positions(ac_l[nearest_idx(ac_l_t, t)][1])
                ar = extract_joint_positions(ac_r[nearest_idx(ac_r_t, t)][1])
                action = al + ar

                odom_vec = np.zeros(13, dtype=np.float32)
                if len(odom_t) > 0:
                    om = odom_msgs[nearest_idx(odom_t, t)][1]
                    p = om.pose.pose
                    tw = om.twist.twist
                    odom_vec[0:3] = [p.position.x, p.position.y, p.position.z]
                    odom_vec[3:7] = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
                    odom_vec[7:10] = [tw.linear.x, tw.linear.y, tw.linear.z]
                    odom_vec[10:13] = [tw.angular.x, tw.angular.y, tw.angular.z]

                frames.append({
                    "observation.images.realsense_top": img_top,
                    "observation.images.realsense_left": img_left,
                    "observation.images.realsense_right": img_right,
                    "observation.state": np.array(state, dtype=np.float32),
                    "observation.odom": odom_vec,
                    "action": np.array(action, dtype=np.float32),
                    "task": task_label,
                })

            elapsed = time.time() - bag_start
            print(f"  Done: {len(frames)} frames in {elapsed:.1f}s")
            return frames if frames else None

    except Exception as e:
        print(f"  Error processing {bag_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# MAIN
# ============================================================
def main():
    total_start = time.time()

    output_path = OUTPUT_DIR / f"{STAGE}_lerobot"
    if output_path.exists():
        shutil.rmtree(output_path)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (ARM_DOF * 2,),
            "names": [f"left_joint_{i}" for i in range(7)]
            + [f"right_joint_{i}" for i in range(7)],
        },
        "action": {
            "dtype": "float32",
            "shape": (ARM_DOF * 2,),
            "names": [f"left_joint_{i}" for i in range(7)]
            + [f"right_joint_{i}" for i in range(7)],
        },
        "observation.odom": {
            "dtype": "float32",
            "shape": (13,),
            "names": [
                "pos_x", "pos_y", "pos_z",
                "quat_x", "quat_y", "quat_z", "quat_w",
                "linear_vx", "linear_vy", "linear_vz",
                "angular_wx", "angular_wy", "angular_wz",
            ],
        },
        "observation.images.realsense_top": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.realsense_left": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.realsense_right": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=f"{STAGE}_lerobot",
        root=output_path,
        robot_type="zeno",
        fps=FPS,
        features=features,
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=4,
    )

    print(f"\n{'=' * 60}")
    print(f"Converting {STAGE} to LeRobot V3.0")
    print(f"  Task:    {TASK_NAME}")
    print(f"  Input:   {DATA_ROOT}")
    print(f"  Output:  {output_path}")
    print(f"  FPS:     {FPS}")
    print(f"  ImgSize: {IMG_SIZE}")
    print(f"{'=' * 60}")

    bag_files = sorted(DATA_ROOT.glob("*.bag"))
    total_bags = len(bag_files)

    if total_bags == 0:
        print("No bag files found.")
        return

    print(f"Found {total_bags} bag files")

    successful = 0
    for bag_idx, bag_path in enumerate(bag_files, 1):
        result = process_single_bag(bag_path, bag_idx, total_bags, task_label=TASK_NAME)
        if result is not None:
            for frame in result:
                dataset.add_frame(frame)
            dataset.save_episode()
            successful += 1
            print(f"  [Bag {bag_idx}/{total_bags}] Saved episode.")
        del result

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
