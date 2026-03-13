#!/usr/bin/env python3
"""
MoE-ACT 模型部署 - 单一统一模型自动路由
MoE-ACT Policy Deployment with Automatic Expert Routing

关键特性：
1. 单一 MoE-ACT 模型：内置 10 个 expert，自动路由到合适的 expert
2. 统一归一化：使用训练时的归一化参数
3. 无需分类器：模型自动学习任务分布
"""
import argparse
from pathlib import Path
import importlib.util

import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, JointState
import cv2
import torch
from PIL import Image

from lerobot.policies.moe_act.modeling_moe_act import MoEACTPolicy
from topic_tools.srv import MuxSelect
from std_msgs.msg import Bool
from pynput import keyboard


# ====== 全局变量 ======
# 当前激活的模型和归一化参数
current_policy = None
current_norm_params = None

# 图像数据缓存
latest_imgs = {
    "fisheye_left": None,
    "fisheye_right": None,
    "wide_mid": None,
    "wide_top": None,
}

# 关节状态缓存
latest_q = {
    "left": None,
    "right": None,
}

# 平滑动作缓存
smoothed_action = {
    "left": None,
    "right": None,
}

# 其他全局变量
success_key = False
device = None


# ====== 归一化参数加载 ======
def load_normalization_params(norm_file):
    """加载归一化参数

    Args:
        norm_file: 归一化参数文件路径

    Returns:
        dict: 包含STATE_MEAN, STATE_STD, ACTION_MEAN, ACTION_STD等参数的字典
    """
    try:
        spec = importlib.util.spec_from_file_location("norm_module", norm_file)
        norm_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(norm_module)

        params = {
            'STATE_MEAN': norm_module.STATE_MEAN,
            'STATE_STD': norm_module.STATE_STD,
            'ACTION_MEAN': norm_module.ACTION_MEAN,
            'ACTION_STD': norm_module.ACTION_STD,
            'IMAGE_MEAN': norm_module.IMAGE_MEAN,
            'IMAGE_STD': norm_module.IMAGE_STD,
        }

        rospy.loginfo(f"成功加载归一化参数: {norm_file}")
        return params

    except Exception as e:
        rospy.logerr(f"加载归一化参数失败: {e}")
        raise


# ====== 模型加载 ======
def load_moe_act_policy(model_path, device_str):
    """加载 MoE-ACT 策略模型

    Args:
        model_path: 模型路径
        device_str: 设备 ('cuda' 或 'cpu')

    Returns:
        MoEACTPolicy: 加载的策略模型
    """
    try:
        local_path = Path(model_path).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        rospy.loginfo(f"正在加载 MoE-ACT 模型: {model_path}")

        policy = MoEACTPolicy.from_pretrained(pretrained_name_or_path=str(local_path))
        policy = policy.to(device_str)
        policy.eval()

        rospy.loginfo("=" * 70)
        rospy.loginfo("[INFO] MoE-ACT 模型加载成功")
        rospy.loginfo(f"  chunk_size: {policy.config.chunk_size}")
        rospy.loginfo(f"  n_action_steps: {policy.config.n_action_steps}")
        rospy.loginfo(f"  num_experts: {policy.config.num_experts}")
        rospy.loginfo(f"  num_experts_per_token: {policy.config.num_experts_per_token}")
        rospy.loginfo("=" * 70)

        return policy

    except Exception as e:
        rospy.logerr(f"加载模型失败: {e}")
        raise


# ====== ROS 回调函数 ======
def decode_compressed_image(msg: CompressedImage) -> np.ndarray:
    """解码压缩图像消息为 BGR numpy 数组"""
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img_bgr


def cb_fisheye_left(msg: CompressedImage):
    latest_imgs["fisheye_left"] = decode_compressed_image(msg)

def cb_fisheye_right(msg: CompressedImage):
    latest_imgs["fisheye_right"] = decode_compressed_image(msg)

def cb_wide_mid(msg: CompressedImage):
    latest_imgs["wide_mid"] = decode_compressed_image(msg)

def cb_wide_top(msg: CompressedImage):
    latest_imgs["wide_top"] = decode_compressed_image(msg)

def cb_joints_left(msg: JointState):
    latest_q["left"] = np.array(msg.position, dtype=np.float32)

def cb_joints_right(msg: JointState):
    latest_q["right"] = np.array(msg.position, dtype=np.float32)


# ====== 预处理函数 ======
def preprocess_image(img_bgr: np.ndarray, norm_params: dict) -> torch.Tensor:
    """Convert BGR uint8 (H,W,3) to normalized float32 torch tensor (1,3,256,256)."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - norm_params['IMAGE_MEAN']) / norm_params['IMAGE_STD']
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return torch.from_numpy(img)


# 静止臂关节的 std 趋近于零，导致归一化/反归一化产生极端值。
# 训练时静止臂输入恒为 ~0（因为 value ≈ mean），部署时传感器微小噪声
# 会被放大为巨大的归一化值。用 clamp 保证部署时的数值稳定性。
STD_CLAMP_MIN = 0.01


def normalize_state(state: np.ndarray, norm_params: dict) -> np.ndarray:
    """使用指定的归一化参数对状态进行归一化，clamp std 避免静止臂除零"""
    std_safe = np.maximum(norm_params['STATE_STD'], STD_CLAMP_MIN)
    return (state - norm_params['STATE_MEAN']) / std_safe


def unnormalize_action(action: np.ndarray, norm_params: dict) -> np.ndarray:
    """使用指定的归一化参数对动作进行反归一化，clamp std 避免静止臂坍塌"""
    std_safe = np.maximum(norm_params['ACTION_STD'], STD_CLAMP_MIN)
    return action * std_safe + norm_params['ACTION_MEAN']


# ====== 键盘监听器 ======
def on_press(key):
    """键盘按键处理"""
    global success_key
    try:
        if key == keyboard.Key.space:
            success_key = True
        elif hasattr(key, 'char'):
            if key.char == "c":
                # 切换遥操模式
                current_mode = rospy.get_param("/is_teleop", False)
                rospy.set_param("/is_teleop", not current_mode)
                rospy.loginfo(f"遥操模式切换: {not current_mode}")
    except AttributeError:
        pass


def main():
    global current_policy, current_norm_params
    global device, smoothed_action

    # 初始化 ROS 节点
    rospy.init_node("hanger_moe_act_deploy")
    rospy.set_param("/is_teleop", False)

    # 等待服务
    rospy.wait_for_service('/robot/arm_left/joint_cmd_mux_select')
    rospy.wait_for_service('/robot/arm_right/joint_cmd_mux_select')
    select_left = rospy.ServiceProxy('/robot/arm_left/joint_cmd_mux_select', MuxSelect)
    select_right = rospy.ServiceProxy('/robot/arm_right/joint_cmd_mux_select', MuxSelect)
    select_left('/robot/arm_left/vla_joint_cmd')
    select_right('/robot/arm_right/vla_joint_cmd')

    master_flag_topic = rospy.Publisher(
        "/conrft_robot/slave_follow_flag",
        Bool,
        queue_size=1,
    )
    Replaced = False
    master_flag_topic.publish(Bool(not Replaced))

    # 启动键盘监听器
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MoE-ACT 策略部署")
    parser.add_argument("--model-path", type=str,
                        default="/workspace/Mult-skill ACT/models/26-03-04-pick_hanger_V30/checkpoints/100010/pretrained_model",
                        help="MoE-ACT 模型路径")
    parser.add_argument("--norm-file", type=str,
                        default="/workspace/Mult-skill ACT/scripts/normalization/26-03-04-pick_hanger_V30_norm.py",
                        help="归一化参数文件路径")
    parser.add_argument("--rate", type=float, default=10.0,
                        help="控制频率 Hz (default: 10)")
    parser.add_argument("--smoothing", type=float, default=0.3,
                        help="EMA平滑系数 (0=无平滑, 1=无历史)")
    parser.add_argument("--no-smoothing", action="store_true",
                        help="禁用EMA平滑")
    parser.add_argument("--action-steps", type=int, default=None,
                        help="重新推理前的动作步数")

    # ROS话题配置
    parser.add_argument("--topic-fisheye-left", type=str,
                        default="/fisheye_left/image_raw/compressed")
    parser.add_argument("--topic-fisheye-right", type=str,
                        default="/fisheye_right/image_raw/compressed")
    parser.add_argument("--topic-wide-mid", type=str,
                        default="/wide_mid/image_raw/compressed")
    parser.add_argument("--topic-wide-top", type=str,
                        default="/wide_top/image_raw/compressed")
    parser.add_argument("--topic-joint-left", type=str,
                        default="/robot/arm_left/joint_states_single")
    parser.add_argument("--topic-joint-right", type=str,
                        default="/robot/arm_right/joint_states_single")
    parser.add_argument("--topic-cmd-left", type=str,
                        default="/robot/arm_left/vla_joint_cmd")
    parser.add_argument("--topic-cmd-right", type=str,
                        default="/robot/arm_right/vla_joint_cmd")

    args, _ = parser.parse_known_args()

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rospy.loginfo(f"使用设备: {device}")

    # 加载归一化参数
    rospy.loginfo("正在加载归一化参数...")
    current_norm_params = load_normalization_params(args.norm_file)

    # 加载 MoE-ACT 模型
    rospy.loginfo("正在加载 MoE-ACT 模型...")
    current_policy = load_moe_act_policy(args.model_path, device)

    # 覆盖 n_action_steps 配置
    if args.action_steps is not None:
        current_policy.config.n_action_steps = args.action_steps
        rospy.loginfo(f"覆盖 n_action_steps 为: {args.action_steps}")

    current_policy.reset()

    # ROS 订阅器
    rospy.Subscriber(args.topic_fisheye_left, CompressedImage, cb_fisheye_left, queue_size=1)
    rospy.Subscriber(args.topic_fisheye_right, CompressedImage, cb_fisheye_right, queue_size=1)
    rospy.Subscriber(args.topic_wide_mid, CompressedImage, cb_wide_mid, queue_size=1)
    rospy.Subscriber(args.topic_wide_top, CompressedImage, cb_wide_top, queue_size=1)
    rospy.Subscriber(args.topic_joint_left, JointState, cb_joints_left, queue_size=1)
    rospy.Subscriber(args.topic_joint_right, JointState, cb_joints_right, queue_size=1)

    # ROS 发布器
    pub_left = rospy.Publisher(args.topic_cmd_left, JointState, queue_size=1)
    pub_right = rospy.Publisher(args.topic_cmd_right, JointState, queue_size=1)

    rate = rospy.Rate(args.rate)

    ENABLE_SMOOTHING = not args.no_smoothing
    SMOOTHING_ALPHA = args.smoothing

    rospy.loginfo("=" * 70)
    rospy.loginfo("[CONFIG] 部署设置:")
    rospy.loginfo(f"  模型: MoE-ACT ({current_policy.config.num_experts} experts)")
    rospy.loginfo(f"  控制频率: {args.rate} Hz")
    rospy.loginfo(f"  EMA平滑: {ENABLE_SMOOTHING}, alpha={SMOOTHING_ALPHA}")
    rospy.loginfo("  按 'c' 键切换遥操模式")
    rospy.loginfo("=" * 70)
    rospy.loginfo("等待传感器数据...")

    data_ready_logged = False
    step_count = 0

    while not rospy.is_shutdown():
        # 检查必要数据是否就绪
        if (latest_imgs["fisheye_left"] is None or
            latest_imgs["fisheye_right"] is None or
            latest_imgs["wide_mid"] is None or
            latest_imgs["wide_top"] is None):
            rate.sleep()
            continue

        if latest_q["left"] is None or latest_q["right"] is None:
            rate.sleep()
            continue

        if not data_ready_logged:
            rospy.loginfo("所有传感器就绪，开始推理...")
            data_ready_logged = True

        # 构建观测字典
        img_fisheye_left = preprocess_image(latest_imgs["fisheye_left"], current_norm_params).to(device)
        img_fisheye_right = preprocess_image(latest_imgs["fisheye_right"], current_norm_params).to(device)
        img_wide_mid = preprocess_image(latest_imgs["wide_mid"], current_norm_params).to(device)
        img_wide_top = preprocess_image(latest_imgs["wide_top"], current_norm_params).to(device)

        # State 归一化 (14D)
        state_raw = np.concatenate([latest_q["left"], latest_q["right"]], axis=0).astype(np.float32)
        state_normalized = normalize_state(state_raw, current_norm_params)
        state = torch.from_numpy(state_normalized[None, :]).to(device)

        # 构建观测字典
        obs = {
            "observation.images.fisheye_left": img_fisheye_left,
            "observation.images.fisheye_right": img_fisheye_right,
            "observation.images.wide_mid": img_wide_mid,
            "observation.images.wide_top": img_wide_top,
            "observation.state": state,
        }

        # 模型推理
        with torch.no_grad():
            action_tensor = current_policy.select_action(obs)

        if action_tensor.dim() == 2:
            action_normalized = action_tensor[0, :].cpu().numpy()
        else:
            action_normalized = action_tensor.cpu().numpy()

        # Action 反归一化 (14D)
        action = unnormalize_action(action_normalized, current_norm_params)

        if len(action) != 14:
            rospy.logwarn(f"无效的动作维度: {len(action)}, 期望 14")
            rate.sleep()
            continue

        # 拆分 action: [left_arm×7, right_arm×7]
        action_left = action[0:7].copy()
        action_right = action[7:14].copy()

        # 诊断日志
        step_count += 1
        if step_count <= 5 or step_count % 50 == 0:
            rospy.loginfo("=" * 70)
            rospy.loginfo(f"[DIAG] Step {step_count} | MoE-ACT")
            rospy.loginfo(f"  Raw state LEFT:        {np.array2string(latest_q['left'], precision=3)}")
            rospy.loginfo(f"  Raw state RIGHT:       {np.array2string(latest_q['right'], precision=3)}")
            rospy.loginfo(f"  Model output (norm):   {np.array2string(action_normalized[:5], precision=3)}...")
            rospy.loginfo(f"  Unnorm action LEFT:    {np.array2string(action_left, precision=3)}")
            rospy.loginfo(f"  Unnorm action RIGHT:   {np.array2string(action_right, precision=3)}")

            delta_left = action_left - latest_q["left"][:7]
            delta_right = action_right - latest_q["right"][:7]
            rospy.loginfo(f"  Delta LEFT:            {np.array2string(delta_left, precision=3)}")
            rospy.loginfo(f"  Delta RIGHT:           {np.array2string(delta_right, precision=3)}")
            rospy.loginfo("=" * 70)

        # EMA 平滑
        if ENABLE_SMOOTHING:
            if smoothed_action["left"] is None:
                smoothed_action["left"] = action_left
                smoothed_action["right"] = action_right
            else:
                smoothed_action["left"] = SMOOTHING_ALPHA * action_left + (1.0 - SMOOTHING_ALPHA) * smoothed_action["left"]
                smoothed_action["right"] = SMOOTHING_ALPHA * action_right + (1.0 - SMOOTHING_ALPHA) * smoothed_action["right"]
            action_left = smoothed_action["left"]
            action_right = smoothed_action["right"]

        # 人工干预判断
        Replaced = rospy.get_param("/is_teleop", False)
        master_flag_topic.publish(Bool(not Replaced))

        if Replaced:
            select_left('/teleop/arm_left/joint_states_single')
            select_right('/teleop/arm_right/joint_states_single')
        else:
            select_left('/robot/arm_left/vla_joint_cmd')
            select_right('/robot/arm_right/vla_joint_cmd')

        # 发布控制命令
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

        # 左臂关节命令
        msg_left = JointState()
        msg_left.header.stamp = rospy.Time.now()
        msg_left.name = joint_names
        msg_left.position = action_left.tolist()
        if not Replaced:
            pub_left.publish(msg_left)

        # 右臂关节命令
        msg_right = JointState()
        msg_right.header.stamp = rospy.Time.now()
        msg_right.name = joint_names
        msg_right.position = action_right.tolist()
        if not Replaced:
            pub_right.publish(msg_right)

        rospy.loginfo_throttle(2.0, f"Actions sent - Step {step_count} | MoE-ACT")

        rate.sleep()


if __name__ == "__main__":
    main()
