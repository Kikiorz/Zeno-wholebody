#!/usr/bin/env python3
"""
SARM-Gated Dual-Expert Deployment for Zeno Dual-Arm Robot.

Replaces the custom TemporalProgressRouter with SARM for stage switching.
SARM runs ~1x/second (frame_gap=30 at 30Hz) using CLIP ViT-B/32 features
and 9-frame bidirectional temporal context.

Switch condition: P(stage="move_hanger") > threshold for N consecutive SARM preds.

Control frequency: 30Hz (ACT runs every tick, SARM runs every ~30 ticks)
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image
from sensor_msgs.msg import CompressedImage, JointState
from transformers import CLIPModel, CLIPProcessor

# ── Project paths ──
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lerobot" / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.sarm.modeling_sarm import SARMRewardModel
from lerobot.policies.sarm.sarm_utils import pad_state_to_max_dim


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
    0.13790602308137262, 1.1934556628106265, 0.8850414156913757, 0.25022798776626587,
    0.22821392118930817, 0.7877830862998962, 0.04246484860777855,
    0.1858433187007904, 0.8290647268295288, 0.591930091381073, 0.579460620880127,
    0.32592472434043884, 0.8642625212669373, 0.043366722762584686
], dtype=np.float32)

S2_ACTION_MEAN = np.array([
    0.02281772904098034, -0.045251425355672836, 0.0,
    -0.1395794302225113, -0.044002238661050797, -0.44064539670944214, 0.08966775238513947,
    0.9203009605407715, -0.0745759904384613, 0.05977959930896759,
    0.18610866367816925, 0.8303110003471375, -1.0536752939224243, -0.1983378827571869,
    0.9438851475715637, 0.14664919674396515, 0.02227647230029106
], dtype=np.float32)

S2_ACTION_STD = np.array([
    0.0001591314357938245, 1e-6, 1e-6,
    0.13790602308137262, 1.1934556628106265, 0.8850414156913757, 0.25022798776626587,
    0.22821392118930817, 0.7877830862998962, 0.04246484860777855,
    0.1858433187007904, 0.8290647268295288, 0.591930091381073, 0.579460620880127,
    0.32592472434043884, 0.8642625212669373, 0.043366722762584686
], dtype=np.float32)


# ====== SARM Inference Helper ======

class SARMSwitchDetector:
    """Manages SARM inference for stage A→B switching.

    Maintains a ring buffer of CLIP image embeddings at 1s intervals (frame_gap=30).
    Runs SARM once per second and tracks consecutive stage-B predictions.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        stage_b_threshold: float = 0.8,
        confirm_count: int = 3,
        task_description: str = "hang clothes then move hanger",
    ):
        # Load SARM model
        self.sarm = SARMRewardModel.from_pretrained(model_path)
        self.sarm.config.device = device
        self.sarm.to(device).eval()
        self.device = torch.device(device)

        self.n_obs_steps = self.sarm.config.n_obs_steps  # 8
        self.frame_gap = self.sarm.config.frame_gap  # 30
        self.n_obs_frames = 1 + self.n_obs_steps  # 9 (bidirectional: 4 past + current + 4 future)

        # CLIP encoder
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

        # Pre-encode task text
        with torch.no_grad():
            text_inputs = self.clip_processor.tokenizer(
                [task_description], return_tensors="pt", padding=True, truncation=True
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            self.text_emb = self.clip_model.get_text_features(**text_inputs).detach()  # (1, 512)

        # Ring buffer for CLIP embeddings at 1s intervals
        # We store embeddings for frames at t, t-30, t-60, ..., t-240 (9 frames)
        # But SARM uses bidirectional: [-120, -90, -60, -30, 0, +30, +60, +90, +120]
        # At deployment we only have past+current, so we fill future slots with current frame
        self.emb_buffer = deque(maxlen=self.n_obs_frames)
        self.state_buffer = deque(maxlen=self.n_obs_frames)

        # Switch detection
        self.stage_b_threshold = stage_b_threshold
        self.confirm_count = confirm_count
        self.consecutive_b = 0
        self.switched = False
        self.tick = 0

        # Dense head has 2 stages: hang_clothes (0), move_hanger (1)
        self.dense_stage_idx_b = 1

    @torch.no_grad()
    def _encode_image(self, img_bgr: np.ndarray) -> torch.Tensor:
        """Encode a single BGR image to CLIP embedding (512,)."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        inputs = self.clip_processor(images=[pil_img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.clip_model.get_image_features(**inputs).detach()  # (1, 512)
        return emb.squeeze(0)  # (512,)

    def update(self, img_bgr: np.ndarray, state: np.ndarray) -> dict:
        """Called every tick (30Hz). Encodes image every frame_gap ticks and runs SARM.

        Returns dict with:
            - 'should_switch': bool — True when switch condition met (one-time)
            - 'stage_b_prob': float — P(move_hanger) from latest SARM prediction
            - 'ran_sarm': bool — whether SARM was invoked this tick
        """
        self.tick += 1
        result = {"should_switch": False, "stage_b_prob": 0.0, "ran_sarm": False}

        if self.switched:
            return result

        # Encode and buffer every frame_gap ticks (1s)
        if self.tick % self.frame_gap == 1 or self.tick == 1:
            emb = self._encode_image(img_bgr)
            self.emb_buffer.append(emb)

            # Pad state to max_state_dim=32
            state_t = torch.tensor(state, dtype=torch.float32)
            if state_t.shape[0] < self.sarm.config.max_state_dim:
                state_t = torch.nn.functional.pad(
                    state_t, (0, self.sarm.config.max_state_dim - state_t.shape[0])
                )
            self.state_buffer.append(state_t)

        # Need at least a few frames before running SARM
        if len(self.emb_buffer) < 3:
            return result

        # Run SARM once per second (same cadence as buffering)
        if self.tick % self.frame_gap != 1 and self.tick != 1:
            return result

        result["ran_sarm"] = True

        # Build input tensors
        # Pad to n_obs_frames if we don't have enough history yet
        emb_list = list(self.emb_buffer)
        state_list = list(self.state_buffer)
        n_have = len(emb_list)

        # Pad by repeating earliest frame
        while len(emb_list) < self.n_obs_frames:
            emb_list.insert(0, emb_list[0])
            state_list.insert(0, state_list[0])

        video_emb = torch.stack(emb_list).unsqueeze(0)  # (1, 9, 512)
        state_feat = torch.stack(state_list).unsqueeze(0)  # (1, 9, 32)
        text_emb = self.text_emb  # (1, 512)
        lengths = torch.tensor([min(n_have, self.n_obs_frames)], dtype=torch.int32)

        # Run SARM with dense head (2 stages: hang_clothes, move_hanger)
        rewards, stage_probs = self.sarm.calculate_rewards(
            text_embeddings=text_emb,
            video_embeddings=video_emb,
            state_features=state_feat,
            lengths=lengths,
            return_stages=True,
            head_mode="dense",
        )

        # stage_probs shape: (T, 2) for single sample — extract center frame
        center_idx = min(self.n_obs_steps // 2, stage_probs.shape[0] - 1)
        p_b = float(stage_probs[center_idx, self.dense_stage_idx_b])
        result["stage_b_prob"] = p_b

        # Confirm logic
        if p_b > self.stage_b_threshold:
            self.consecutive_b += 1
        else:
            self.consecutive_b = 0

        if self.consecutive_b >= self.confirm_count:
            self.switched = True
            result["should_switch"] = True

        return result


# ====== ROS Sensor State ======

latest_images = {}
latest_state = None
latest_odom = None


def cb_image(msg, cam_name):
    global latest_images
    arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        latest_images[cam_name] = cv2.resize(img, (224, 224))


def cb_joint(msg):
    global latest_state
    latest_state = np.array(msg.position, dtype=np.float32)


def cb_odom(msg):
    global latest_odom
    latest_odom = msg


def get_state_17d():
    """Compose 17D state: [base_vx, base_vy, base_omega, left_7, right_7]."""
    if latest_state is None or latest_odom is None:
        return None
    vx = latest_odom.twist.twist.linear.x
    vy = latest_odom.twist.twist.linear.y
    wz = latest_odom.twist.twist.angular.z
    return np.concatenate([[vx, vy, wz], latest_state[:14]], dtype=np.float32)


# ====== ACT helpers ======

def load_act(ckpt_path, device):
    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.to(device).eval()
    return policy


def act_inference(policy, images, state, state_mean, state_std, action_mean, action_std, device):
    """Run ACT inference with manual normalization (same as run_router_switch.py)."""
    state_norm = (state - state_mean) / state_std

    obs = {"observation.state": torch.from_numpy(state_norm).float().unsqueeze(0).to(device)}
    for cam_name, img in images.items():
        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        obs[f"observation.images.{cam_name}"] = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        action_norm = policy.select_action(obs).cpu().numpy().squeeze(0)

    action = action_norm * action_std + action_mean
    return action


# ====== Main ======

def main():
    parser = argparse.ArgumentParser(description="SARM-gated dual-expert deployment")
    parser.add_argument("--stage1-ckpt", type=str, required=True, help="Stage A (hang) ACT checkpoint")
    parser.add_argument("--stage2-ckpt", type=str, required=True, help="Stage B (move) ACT checkpoint")
    parser.add_argument("--sarm-ckpt", type=str, required=True, help="SARM model checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.8, help="P(stage_B) threshold")
    parser.add_argument("--confirm", type=int, default=3, help="Consecutive SARM confirmations needed")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device

    # Load models
    rospy.loginfo("Loading Stage A ACT...")
    policy_a = load_act(args.stage1_ckpt, device)

    rospy.loginfo("Loading Stage B ACT...")
    policy_b = load_act(args.stage2_ckpt, device)

    rospy.loginfo("Loading SARM switch detector...")
    sarm_detector = SARMSwitchDetector(
        model_path=args.sarm_ckpt,
        device=device,
        stage_b_threshold=args.threshold,
        confirm_count=args.confirm,
    )

    # ROS setup
    rospy.init_node("sarm_switch_controller", anonymous=True)

    CAMERAS = {
        "realsense_top": "/realsense_top/color/image_raw/compressed",
        "realsense_left": "/realsense_left/color/image_raw/compressed",
        "realsense_right": "/realsense_right/color/image_raw/compressed",
    }
    for cam_name, topic in CAMERAS.items():
        rospy.Subscriber(topic, CompressedImage, cb_image, cam_name)

    rospy.Subscriber("/joint_states", JointState, cb_joint)
    rospy.Subscriber("/odom", Odometry, cb_odom)

    pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    pub_left = rospy.Publisher("/left_arm/joint_command", JointState, queue_size=1)
    pub_right = rospy.Publisher("/right_arm/joint_command", JointState, queue_size=1)

    rate = rospy.Rate(30)
    rospy.loginfo("Waiting for sensor data...")
    while not rospy.is_shutdown():
        if len(latest_images) == 3 and latest_state is not None and latest_odom is not None:
            break
        rate.sleep()

    rospy.loginfo("Starting SARM-gated control loop")
    active_stage = "A"
    tick = 0

    while not rospy.is_shutdown():
        tick += 1
        state_17d = get_state_17d()
        if state_17d is None:
            rate.sleep()
            continue

        images = dict(latest_images)
        if len(images) < 3:
            rate.sleep()
            continue

        # ── SARM switch detection (uses realsense_top only) ──
        if active_stage == "A":
            sarm_result = sarm_detector.update(images["realsense_top"], state_17d)

            if sarm_result["ran_sarm"]:
                rospy.loginfo(
                    f"[tick={tick}] SARM: P(move_hanger)={sarm_result['stage_b_prob']:.3f} "
                    f"consec={sarm_detector.consecutive_b}/{args.confirm}"
                )

            if sarm_result["should_switch"]:
                active_stage = "B"
                rospy.logwarn(f"[tick={tick}] === SWITCHING A → B ===")

        # ── ACT inference (every tick at 30Hz) ──
        if active_stage == "A":
            action = act_inference(
                policy_a, images, state_17d,
                S1_STATE_MEAN, S1_STATE_STD, S1_ACTION_MEAN, S1_ACTION_STD, device,
            )
        else:
            action = act_inference(
                policy_b, images, state_17d,
                S2_STATE_MEAN, S2_STATE_STD, S2_ACTION_MEAN, S2_ACTION_STD, device,
            )

        # ── Parse action: [base_vx, base_vy, base_wz, left_7, right_7] ──
        action_base = action[:3]
        action_left = action[3:10]
        action_right = action[10:17]

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
