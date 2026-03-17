#!/usr/bin/env python3
"""
Router-Gated Dual-Expert Deployment for Zeno Dual-Arm Robot.

Runs stageA (hanger) and stageB (move) ACT policies simultaneously.
A TemporalProgressRouter predicts when to perform a ONE-TIME switch from A→B.

Signals:
  - START: first observation → if router p[0] > 0.5 → start with B, else A
  - SWITCH: crossing_idx countdown reaches 0 → permanently switch to B

Control frequency: 30Hz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import cv2
import torch

# ── Project paths ──
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lerobot" / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.policies.act.modeling_act import ACTPolicy
from router.model import TemporalProgressRouter, RouterConfig


# ====== Normalization statistics ======

# --- Stage1 (hanger) ---
S1_STATE_MEAN = np.array([
    5.034777359469444e-07, 0.0, 0.0,
    -0.04810221865773201, 1.3781161308288574, -1.3469257354736328, -0.16103820502758026,
    0.9833335280418396, 0.4252128005027771, 0.04083471745252609,
    0.20126894116401672, 2.0605885982513428, -1.7873194217681885, 0.5087009072303772,
    0.9461426734924316, -1.0566874742507935, 0.02339804731309414
], dtype=np.float32)

S1_STATE_STD = np.array([
    0.08398732542991638, 0.0863003060221672, 1e-6,
    0.13790602308137262, 1.1934556628106265, 0.8850414156913757, 0.25022798776626587,
    0.22821392118930817, 0.7877830862998962, 0.04246484860777855,
    0.1858433187007904, 0.8290647268295288, 0.591930091381073, 0.579460620880127,
    0.32592472434043884, 0.8642625212669373, 0.043366722762584686
], dtype=np.float32)

S1_ACTION_MEAN = np.array([
    5.034777359469444e-07, 0.0, 0.0,
    -0.046814993023872375, 1.353559970855713, -1.3664793968200684, -0.1808137744665146,
    0.9312586188316345, 0.42406341433525085, 0.04844813048839569,
    0.20318368077278137, 2.018932342529297, -1.809546709060669, 0.5431661009788513,
    0.9190205931663513, -1.058423399925232, 0.01156390830874443
], dtype=np.float32)

S1_ACTION_STD = np.array([
    0.0001591314357938245, 1e-6, 1e-6,
    0.1379060298204422, 1.193455696105957, 0.8850414156913757, 0.25022798776626587,
    0.22821392118930817, 0.7877830862998962, 0.04246484860777855,
    0.1858433187007904, 0.8290647268295288, 0.591930091381073, 0.579460620880127,
    0.32592472434043884, 0.8642625212669373, 0.043366722762584686
], dtype=np.float32)

# --- Stage2 (move) ---
S2_STATE_MEAN = np.array([
    0.02281772904098034, -0.045251425355672836, 0.0,
    -0.1395794302225113, -0.044002238661050797, -0.44064539670944214, 0.08966775238513947,
    0.9203009605407715, -0.0745759904384613, 0.05977959930896759,
    0.18610866367816925, 0.8303110003471375, -1.0536752939224243, -0.1983378827571869,
    0.9438851475715637, 0.14664919674396515, 0.02227647230029106
], dtype=np.float32)

S2_STATE_STD = np.array([
    0.08398732542991638, 0.0863003060221672, 1e-6,
    0.0010548126883804798, 0.0006011321092955768, 0.0103145157918334, 0.049441609531641006,
    0.020588429644703865, 0.014086958020925522, 0.00018027082842309028,
    0.2019774466753006, 0.4483957588672638, 0.4555847942829132, 0.192733496427536,
    0.21775825321674347, 0.26127147674560547, 0.027230218052864075
], dtype=np.float32)

S2_ACTION_MEAN = np.array([
    0.02281772904098034, -0.045251425355672836, 0.0,
    -0.1378067582845688, -0.048895981162786484, -0.46044376492500305, 0.09704867750406265,
    0.8196730017662048, -0.07550995796918869, 0.0802001804113388,
    0.18711987137794495, 0.8261152505874634, -1.0771400928497314, -0.22274112701416016,
    0.8373763561248779, 0.1456107497215271, 0.007241050712764263
], dtype=np.float32)

S2_ACTION_STD = np.array([
    0.08398732542991638, 0.0863003060221672, 1e-6,
    0.0005220270832069218, 5.253412282968384e-08, 0.010607271455228329, 0.05429357662796974,
    0.022038497030735016, 0.014269720762968063, 1.8852814864800393e-09,
    0.20295672118663788, 0.43815308809280396, 0.45726028084754944, 0.2143455147743225,
    0.23055443167686462, 0.26300135254859924, 0.04293931648135185
], dtype=np.float32)

# --- Stage3 / Router ---
ROUTER_STATE_MEAN = np.array([
    0.0273, -0.0006, -0.1023,
    0.0081, 0.8091, -0.9350, -0.1055,
    0.9227, 0.2704, 0.0468,
    0.1395, 1.4978, -1.4257, 0.2035,
    0.8924, -0.5760, 0.0214
], dtype=np.float32)

ROUTER_STATE_STD = np.array([
    0.0786, 0.0113, 0.3254,
    0.1080, 1.1333, 0.7715, 0.2090,
    0.1449, 0.6665, 0.0212,
    0.1709, 0.9523, 0.6765, 0.5133,
    0.2436, 0.8505, 0.0284
], dtype=np.float32)

# Image normalization (ImageNet)
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ====== Default model paths ======
DEFAULT_S1_PATH = PROJECT_ROOT / "model" / "stage1_act" / "checkpoints" / "040000" / "pretrained_model"
DEFAULT_S2_PATH = PROJECT_ROOT / "model" / "stage2_act" / "checkpoints" / "040000" / "pretrained_model"
DEFAULT_ROUTER_PATH = PROJECT_ROOT / "model" / "router" / "checkpoints" / "step_100000.pt"


# ====== Global data cache ======
latest_imgs = {
    "realsense_top": None,
    "realsense_left": None,
    "realsense_right": None,
}
latest_q = {"left": None, "right": None}
latest_base_velocity = None
smoothed_action = {"left": None, "right": None, "base": None}


# ====== Helper / callbacks ======
def decode_compressed_image(msg: CompressedImage) -> np.ndarray:
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def cb_realsense_top(msg):   latest_imgs["realsense_top"] = decode_compressed_image(msg)
def cb_realsense_left(msg):  latest_imgs["realsense_left"] = decode_compressed_image(msg)
def cb_realsense_right(msg): latest_imgs["realsense_right"] = decode_compressed_image(msg)

def cb_joints_left(msg):
    latest_q["left"] = np.array(msg.position, dtype=np.float32)

def cb_joints_right(msg):
    latest_q["right"] = np.array(msg.position, dtype=np.float32)

def cb_odom(msg):
    global latest_base_velocity
    latest_base_velocity = np.array([
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y,
        msg.twist.twist.angular.z,
    ], dtype=np.float32)


# ====== Preprocessing ======
def preprocess_image(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGE_MEAN) / IMAGE_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)  # (1,3,224,224)


def build_state_17d() -> np.ndarray:
    base = latest_base_velocity if latest_base_velocity is not None else np.zeros(3, dtype=np.float32)
    left = latest_q["left"][:7]
    right = latest_q["right"][:7]
    return np.concatenate([base, left, right])  # (17,)


# ====== Model loading ======
def load_act_policy(ckpt_dir: str, device: str, label: str) -> ACTPolicy:
    policy = ACTPolicy.from_pretrained(pretrained_name_or_path=str(ckpt_dir))
    policy = policy.to(device)
    policy.eval()
    rospy.loginfo(f"[{label}] ACT policy loaded from {ckpt_dir}")
    return policy


def load_router(ckpt_path: str, device: str) -> TemporalProgressRouter:
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model = TemporalProgressRouter(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    rospy.loginfo(f"[Router] Loaded from {ckpt_path}")
    return model


# ====== Router inference ======
@torch.no_grad()
def router_predict(router, device):
    """Run router on current observation. Returns (probs, crossing_idx).

    probs: (100,) array of P(stage_B) for each step in the chunk.
    crossing_idx: first index where P(B) > 0.5, or -1 if none.
    """
    # Images for router (same preprocessing as ACT)
    imgs = []
    for cam in ["realsense_top", "realsense_left", "realsense_right"]:
        imgs.append(preprocess_image(latest_imgs[cam]).to(device))

    # State normalized with router (stage3) stats
    state_raw = build_state_17d()
    state_norm = (state_raw - ROUTER_STATE_MEAN) / ROUTER_STATE_STD
    state_t = torch.from_numpy(state_norm).unsqueeze(0).float().to(device)  # (1,17)

    logits = router(imgs, state_t)  # (1, 100)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (100,)

    # crossing_idx: first chunk step where P(B) > 0.5
    above = np.where(probs > 0.5)[0]
    crossing_idx = int(above[0]) if len(above) > 0 else -1

    return probs, crossing_idx


# ====== ACT inference ======
@torch.no_grad()
def act_inference(policy, state_raw, device, s_mean, s_std, a_mean, a_std):
    """Run one ACT inference step, return unnormalized 17D action."""
    state_norm = (state_raw - s_mean) / s_std

    obs = {
        "observation.state": torch.from_numpy(state_norm).unsqueeze(0).float().to(device),
        "observation.images.realsense_top": preprocess_image(latest_imgs["realsense_top"]).to(device),
        "observation.images.realsense_left": preprocess_image(latest_imgs["realsense_left"]).to(device),
        "observation.images.realsense_right": preprocess_image(latest_imgs["realsense_right"]).to(device),
    }

    action_norm = policy.select_action(obs).cpu().numpy().squeeze()  # (17,)
    action_raw = action_norm * a_std + a_mean
    return action_raw


# ====== Main ======
def main():
    parser = argparse.ArgumentParser(description="Router-gated dual-expert deployment")
    parser.add_argument("--ckpt-s1", type=str, default=str(DEFAULT_S1_PATH))
    parser.add_argument("--ckpt-s2", type=str, default=str(DEFAULT_S2_PATH))
    parser.add_argument("--ckpt-router", type=str, default=str(DEFAULT_ROUTER_PATH))
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--smoothing", type=float, default=0.3)
    parser.add_argument("--no-smoothing", action="store_true")
    args, _ = parser.parse_known_args()

    rospy.init_node("zeno_router_switch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rospy.loginfo(f"Device: {device}")

    # Load all three models
    policy_a = load_act_policy(args.ckpt_s1, device, "StageA")
    policy_b = load_act_policy(args.ckpt_s2, device, "StageB")
    router = load_router(args.ckpt_router, device)

    policy_a.reset()
    policy_b.reset()

    # ROS subscribers
    rospy.Subscriber("/realsense_top/color/image_raw/compressed", CompressedImage, cb_realsense_top, queue_size=1)
    rospy.Subscriber("/realsense_left/color/image_raw/compressed", CompressedImage, cb_realsense_left, queue_size=1)
    rospy.Subscriber("/realsense_right/color/image_raw/compressed", CompressedImage, cb_realsense_right, queue_size=1)
    rospy.Subscriber("/robot/arm_left/joint_states_single", JointState, cb_joints_left, queue_size=1)
    rospy.Subscriber("/robot/arm_right/joint_states_single", JointState, cb_joints_right, queue_size=1)
    rospy.Subscriber("/ranger_base_node/odom", Odometry, cb_odom, queue_size=1)

    # ROS publishers
    pub_left = rospy.Publisher("/robot/arm_left/vla_joint_cmd", JointState, queue_size=1)
    pub_right = rospy.Publisher("/robot/arm_right/vla_joint_cmd", JointState, queue_size=1)
    pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    rate = rospy.Rate(args.rate)

    def shutdown_hook():
        rospy.loginfo("Shutting down, sending zero velocity...")
        pub_cmd_vel.publish(Twist())
    rospy.on_shutdown(shutdown_hook)

    ENABLE_SMOOTHING = not args.no_smoothing
    SMOOTHING_ALPHA = args.smoothing

    rospy.loginfo("=" * 70)
    rospy.loginfo("[Router Switch] Waiting for sensor data...")
    rospy.loginfo("=" * 70)

    # ── State machine (minimal: just A or B, one-time switch) ──
    active_expert = None   # "A" or "B", determined by START signal
    switched = False       # True after the one-time A→B switch
    step_count = 0
    crossing_zero_count = 0  # consecutive steps where crossing_idx == 0
    SWITCH_CONFIRM_STEPS = 50  # need 50 consecutive crossing_idx==0 to trigger switch
    global smoothed_action

    while not rospy.is_shutdown():
        # Wait for all sensor data
        if (latest_imgs["realsense_top"] is None or
            latest_imgs["realsense_left"] is None or
            latest_imgs["realsense_right"] is None or
            latest_q["left"] is None or
            latest_q["right"] is None):
            rate.sleep()
            continue

        state_raw = build_state_17d()

        # ── Router inference (every step) ──
        probs, crossing_idx = router_predict(router, device)
        p0 = probs[0]

        # ── START signal: determine initial expert on first observation ──
        if active_expert is None:
            if p0 > 0.5:
                active_expert = "B"
                switched = True  # already in B, no switch needed
                rospy.loginfo(f"[START] p[0]={p0:.3f} > 0.5 → starting with StageB (move)")
            else:
                active_expert = "A"
                rospy.loginfo(f"[START] p[0]={p0:.3f} <= 0.5 → starting with StageA (hanger)")

        # ── SWITCH signal: need 50 consecutive steps with crossing_idx==0 ──
        if not switched and active_expert == "A":
            if crossing_idx == 0:
                crossing_zero_count += 1
            else:
                crossing_zero_count = 0

            if crossing_zero_count >= SWITCH_CONFIRM_STEPS:
                active_expert = "B"
                switched = True
                smoothed_action = {"left": None, "right": None, "base": None}
                policy_b.reset()
                rospy.loginfo("=" * 70)
                rospy.loginfo(
                    f"[SWITCH] crossing_idx==0 confirmed for {SWITCH_CONFIRM_STEPS} steps, "
                    f"p[0]={p0:.3f} → switching to StageB (move)"
                )
                rospy.loginfo("=" * 70)

        # ── ACT inference: only the active expert runs ──
        if active_expert == "A":
            action_raw = act_inference(policy_a, state_raw, device,
                                       S1_STATE_MEAN, S1_STATE_STD,
                                       S1_ACTION_MEAN, S1_ACTION_STD)
        else:
            action_raw = act_inference(policy_b, state_raw, device,
                                       S2_STATE_MEAN, S2_STATE_STD,
                                       S2_ACTION_MEAN, S2_ACTION_STD)

        # Split action
        action_base = action_raw[0:3]
        action_left = action_raw[3:10]
        action_right = action_raw[10:17]

        step_count += 1

        # ── Logging ──
        if step_count % 30 == 1:
            cx_str = str(crossing_idx) if crossing_idx >= 0 else "none"
            rospy.loginfo(
                f"[Step {step_count}] expert={active_expert} switched={switched} "
                f"p[0]={p0:.3f} crossing_idx={cx_str} confirm={crossing_zero_count}/{SWITCH_CONFIRM_STEPS}"
            )

        # ── EMA smoothing ──
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

        # ── Publish control commands ──
        cmd_vel = Twist()
        if abs(action_base[2]) < 0.05:
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

        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

        msg_left = JointState()
        msg_left.header.stamp = rospy.Time.now()
        msg_left.name = joint_names
        msg_left.position = action_left.tolist()
        pub_left.publish(msg_left)

        msg_right = JointState()
        msg_right.header.stamp = rospy.Time.now()
        msg_right.name = joint_names
        msg_right.position = action_right.tolist()
        pub_right.publish(msg_right)

        rate.sleep()


if __name__ == "__main__":
    main()
