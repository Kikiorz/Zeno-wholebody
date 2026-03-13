#!/usr/bin/env python3
"""
ACT Policy Deployment for Zeno Dual-Arm Robot with Mobile Base
Model: stage1_act (task: hanger)

Input:
  - observation.state: 17D [base_vx, base_vy, base_omega, left_arm 7D, right_arm 7D]
  - observation.images.realsense_top: 3x224x224
  - observation.images.realsense_left: 3x224x224
  - observation.images.realsense_right: 3x224x224

Output:
  - action: 17D [base_vx, base_vy, base_omega, left_arm 7D, right_arm 7D]

Control frequency: 30Hz (matches dataset FPS)
"""
import argparse
from pathlib import Path

import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import cv2
import torch

from lerobot.policies.act.modeling_act import ACTPolicy


# ====== Normalization statistics (from stage1 safetensors) ======

# observation.state (17D): [base_vx, base_vy, base_omega, left_7D, right_7D]
STATE_MEAN = np.array([
    5.034777359469444e-07, 0.0, 0.0,
    -0.04810221865773201, 1.3781161308288574, -1.3469257354736328, -0.16103820502758026,
    0.9833335280418396, 0.4252128005027771, 0.04083471745252609,
    0.20126894116401672, 2.0605885982513428, -1.7873194217681885, 0.5087009072303772,
    0.9461426734924316, -1.0566874742507935, 0.02339804731309414
], dtype=np.float32)

STATE_STD = np.array([
    0.08398732542991638, 0.0863003060221672, 1e-6,  # base_omega std=0 → 1e-6
    0.13790602308137262, 1.1934556628106265, 0.8850414156913757, 0.25022798776626587,
    0.22821392118930817, 0.7877830862998962, 0.04246484860777855,
    0.1858433187007904, 0.8290647268295288, 0.591930091381073, 0.579460620880127,
    0.32592472434043884, 0.8642625212669373, 0.043366722762584686
], dtype=np.float32)

# action (17D): [base_vx, base_vy, base_omega, left_7D, right_7D]
ACTION_MEAN = np.array([
    5.034777359469444e-07, 0.0, 0.0,
    -0.046814993023872375, 1.353559970855713, -1.3664793968200684, -0.1808137744665146,
    0.9312586188316345, 0.42406341433525085, 0.04844813048839569,
    0.20318368077278137, 2.018932342529297, -1.809546709060669, 0.5431661009788513,
    0.9190205931663513, -1.058423399925232, 0.01156390830874443
], dtype=np.float32)

ACTION_STD = np.array([
    0.0001591314357938245, 1e-6, 1e-6,  # base_vy, base_omega std=0 → 1e-6
    0.1379060298204422, 1.193455696105957, 0.8850414156913757, 0.25022798776626587,
    0.22821392118930817, 0.7877830862998962, 0.04246484860777855,
    0.1858433187007904, 0.8290647268295288, 0.591930091381073, 0.579460620880127,
    0.32592472434043884, 0.8642625212669373, 0.043366722762584686
], dtype=np.float32)

# Image normalization (ImageNet standard, matches dataset stats)
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ====== Global data cache ======
latest_imgs = {
    "realsense_top": None,
    "realsense_left": None,
    "realsense_right": None,
}

latest_q = {
    "left": None,
    "right": None,
}

latest_base_velocity = None

smoothed_action = {
    "left": None,
    "right": None,
    "base": None,
}


# ====== Helper functions ======
def decode_compressed_image(msg: CompressedImage) -> np.ndarray:
    """Decode compressed image message to BGR numpy array"""
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img_bgr


# ====== ROS callback functions ======
def cb_realsense_top(msg: CompressedImage):
    latest_imgs["realsense_top"] = decode_compressed_image(msg)

def cb_realsense_left(msg: CompressedImage):
    latest_imgs["realsense_left"] = decode_compressed_image(msg)

def cb_realsense_right(msg: CompressedImage):
    latest_imgs["realsense_right"] = decode_compressed_image(msg)

def cb_joints_left(msg: JointState):
    latest_q["left"] = np.array(msg.position, dtype=np.float32)

def cb_joints_right(msg: JointState):
    latest_q["right"] = np.array(msg.position, dtype=np.float32)

def cb_odom(msg: Odometry):
    global latest_base_velocity
    latest_base_velocity = np.array([
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y,
        msg.twist.twist.angular.z
    ], dtype=np.float32)


# ====== Preprocessing functions ======
def preprocess_image(img_bgr: np.ndarray) -> torch.Tensor:
    """Convert BGR uint8 (H,W,3) to normalized float32 torch tensor (1,3,224,224)."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGE_MEAN) / IMAGE_STD
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return torch.from_numpy(img)


def normalize_state(state: np.ndarray) -> np.ndarray:
    return (state - STATE_MEAN) / STATE_STD


def unnormalize_action(action: np.ndarray) -> np.ndarray:
    return action * ACTION_STD + ACTION_MEAN


# ====== Model loading ======
def load_policy(ckpt_dir: str, device: str) -> ACTPolicy:
    local_path = Path(ckpt_dir).expanduser().resolve()
    if local_path.exists():
        pretrained_path = str(local_path)
        rospy.loginfo(f"Loading ACT Policy from local path: {pretrained_path}")
    else:
        pretrained_path = ckpt_dir
        rospy.loginfo(f"Loading ACT Policy from Hugging Face: {pretrained_path}")

    policy = ACTPolicy.from_pretrained(pretrained_name_or_path=pretrained_path)
    policy = policy.to(device)

    rospy.loginfo("=" * 70)
    rospy.loginfo("[INFO] Stage1 Policy loaded successfully")
    rospy.loginfo(f"  chunk_size: {policy.config.chunk_size}")
    rospy.loginfo(f"  n_action_steps: {policy.config.n_action_steps}")
    rospy.loginfo("=" * 70)

    policy.eval()
    return policy


# ====== Default model path ======
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "stage1_act" / "checkpoints" / "040000" / "pretrained_model"


def main():
    parser = argparse.ArgumentParser(description="Stage1 ACT Policy Deployment (hanger)")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_MODEL_PATH),
                        help=f"Checkpoint path (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--rate", type=float, default=30.0,
                        help="Control frequency in Hz (default: 30)")
    parser.add_argument("--smoothing", type=float, default=0.3,
                        help="EMA smoothing alpha (0=no smoothing, 1=no history)")
    parser.add_argument("--no-smoothing", action="store_true",
                        help="Disable EMA smoothing")

    args, _ = parser.parse_known_args()

    ckpt_dir = args.ckpt

    # Initialize ROS node
    rospy.init_node("zeno_act_stage1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rospy.loginfo(f"Using device: {device}")
    rospy.loginfo(f"Model path: {ckpt_dir}")

    # Load model
    policy = load_policy(ckpt_dir, device)
    policy.reset()

    # ====== ROS Subscribers ======
    rospy.Subscriber("/realsense_top/color/image_raw/compressed", CompressedImage, cb_realsense_top, queue_size=1)
    rospy.Subscriber("/realsense_left/color/image_raw/compressed", CompressedImage, cb_realsense_left, queue_size=1)
    rospy.Subscriber("/realsense_right/color/image_raw/compressed", CompressedImage, cb_realsense_right, queue_size=1)
    rospy.Subscriber("/robot/arm_left/joint_states_single", JointState, cb_joints_left, queue_size=1)
    rospy.Subscriber("/robot/arm_right/joint_states_single", JointState, cb_joints_right, queue_size=1)
    rospy.Subscriber("/ranger_base_node/odom", Odometry, cb_odom, queue_size=1)

    # ====== ROS Publishers ======
    pub_left = rospy.Publisher("/robot/arm_left/vla_joint_cmd", JointState, queue_size=1)
    pub_right = rospy.Publisher("/robot/arm_right/vla_joint_cmd", JointState, queue_size=1)
    pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    rate = rospy.Rate(args.rate)

    # Shutdown hook: send zero velocity on exit
    def shutdown_hook():
        rospy.loginfo("Shutting down, sending zero velocity...")
        stop_cmd = Twist()
        pub_cmd_vel.publish(stop_cmd)

    rospy.on_shutdown(shutdown_hook)

    ENABLE_SMOOTHING = not args.no_smoothing
    SMOOTHING_ALPHA = args.smoothing

    rospy.loginfo("=" * 70)
    rospy.loginfo("[CONFIG] Stage1 Deployment settings:")
    rospy.loginfo(f"  Control rate: {args.rate} Hz")
    rospy.loginfo(f"  EMA smoothing: {ENABLE_SMOOTHING}, alpha={SMOOTHING_ALPHA}")
    rospy.loginfo("=" * 70)
    rospy.loginfo("Waiting for sensor data...")

    data_ready_logged = False
    step_count = 0
    global latest_base_velocity

    while not rospy.is_shutdown():
        # Check if required data is ready
        if (latest_imgs["realsense_top"] is None or
            latest_imgs["realsense_left"] is None or
            latest_imgs["realsense_right"] is None):
            rate.sleep()
            continue

        if latest_q["left"] is None or latest_q["right"] is None:
            rate.sleep()
            continue

        if latest_base_velocity is None:
            latest_base_velocity = np.zeros(3, dtype=np.float32)

        if not data_ready_logged:
            rospy.loginfo("All required sensors ready, starting inference...")
            data_ready_logged = True

        # ====== Build observation dictionary ======
        # Image preprocessing
        top_img = preprocess_image(latest_imgs["realsense_top"]).to(device)
        left_img = preprocess_image(latest_imgs["realsense_left"]).to(device)
        right_img = preprocess_image(latest_imgs["realsense_right"]).to(device)

        # State: 17D = [base_vx, base_vy, base_omega, left_7D, right_7D]
        state_raw = np.concatenate([
            latest_base_velocity,
            latest_q["left"][:7],
            latest_q["right"][:7]
        ]).astype(np.float32)
        state_normalized = normalize_state(state_raw)
        state = torch.from_numpy(state_normalized[None, :]).to(device)

        obs = {
            "observation.images.realsense_top": top_img,
            "observation.images.realsense_left": left_img,
            "observation.images.realsense_right": right_img,
            "observation.state": state,
        }

        # ====== Model inference ======
        with torch.no_grad():
            action_tensor = policy.select_action(obs)

        if action_tensor.dim() == 2:
            action_normalized = action_tensor[0, :].cpu().numpy()
        else:
            action_normalized = action_tensor.cpu().numpy()

        # Unnormalize action
        action_raw = unnormalize_action(action_normalized)

        # Split action: [base_3D, left_arm_7D, right_arm_7D]
        action_base = action_raw[0:3]
        action_left = action_raw[3:10]
        action_right = action_raw[10:17]

        step_count += 1

        # Diagnostic logging
        if step_count % 50 == 1:
            rospy.loginfo("=" * 70)
            rospy.loginfo(f"[Step {step_count}] Diagnostic:")
            rospy.loginfo(f"  Raw state:             {np.array2string(state_raw, precision=4)}")
            rospy.loginfo(f"  Action (unnorm) base:  {np.array2string(action_base, precision=4)}")
            rospy.loginfo(f"  Action (unnorm) left:  {np.array2string(action_left, precision=4)}")
            rospy.loginfo(f"  Action (unnorm) right: {np.array2string(action_right, precision=4)}")
            delta_left = action_left - latest_q["left"][:7]
            delta_right = action_right - latest_q["right"][:7]
            rospy.loginfo(f"  Delta LEFT:            {np.array2string(delta_left, precision=3)}")
            rospy.loginfo(f"  Delta RIGHT:           {np.array2string(delta_right, precision=3)}")
            rospy.loginfo("=" * 70)

        # ====== EMA smoothing ======
        global smoothed_action
        if ENABLE_SMOOTHING:
            if smoothed_action["left"] is None:
                smoothed_action["left"] = action_left
                smoothed_action["right"] = action_right
                smoothed_action["base"] = action_base
            else:
                smoothed_action["left"] = SMOOTHING_ALPHA * action_left + (1.0 - SMOOTHING_ALPHA) * smoothed_action["left"]
                smoothed_action["right"] = SMOOTHING_ALPHA * action_right + (1.0 - SMOOTHING_ALPHA) * smoothed_action["right"]
                smoothed_action["base"] = SMOOTHING_ALPHA * action_base + (1.0 - SMOOTHING_ALPHA) * smoothed_action["base"]
            action_left = smoothed_action["left"]
            action_right = smoothed_action["right"]
            action_base = smoothed_action["base"]

        # ====== Publish control commands ======
        # Base velocity command
        cmd_vel = Twist()
        if abs(action_base[2]) < 0.05:  # omega close to 0, forward mode
            cmd_vel.linear.x = float(action_base[0])
            cmd_vel.linear.y = float(action_base[1])
            cmd_vel.angular.z = float(action_base[2])
        elif abs(action_base[2]) >= 0.1:
            cmd_vel.linear.x = 0.0
            cmd_vel.linear.y = float(action_base[0])
            cmd_vel.angular.z = 0.0
        else:
            cmd_vel.linear.x = 0.0
            cmd_vel.linear.y = float(action_base[0])
            cmd_vel.angular.z = 0.0
        pub_cmd_vel.publish(cmd_vel)

        # Left arm joint command
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

        msg_left = JointState()
        msg_left.header.stamp = rospy.Time.now()
        msg_left.name = joint_names
        msg_left.position = action_left.tolist()
        pub_left.publish(msg_left)

        # Right arm joint command
        msg_right = JointState()
        msg_right.header.stamp = rospy.Time.now()
        msg_right.name = joint_names
        msg_right.position = action_right.tolist()
        pub_right.publish(msg_right)

        rospy.loginfo_throttle(2.0, f"Stage1 actions sent (base_vx={action_base[0]:.4f})")

        rate.sleep()


if __name__ == "__main__":
    main()
