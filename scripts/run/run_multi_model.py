#!/usr/bin/env python3
"""
多模型ACT策略部署 - 基于点位分类器的动态模型选择
Multi-Model ACT Policy Deployment with Point Classifier

关键特性：
1. 点位分类器：根据fisheye_left图像分类当前场景（point1-point10）
2. 动态模型加载：根据分类结果加载对应的ACT策略模型
3. 动态归一化：每个点位使用独立的归一化参数
4. 按需触发：通过键盘'p'键触发分类和模型切换
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

from lerobot.policies.act.modeling_act import ACTPolicy
from topic_tools.srv import MuxSelect
from std_msgs.msg import Bool
from pynput import keyboard

# 导入分类器相关函数
from point_classifier_model import PointClassifier
from torchvision import transforms


# ====== 全局变量 ======
# 当前激活的点位和模型
current_point = None
current_policy = None
current_norm_params = None

# 分类器相关
classifier_model = None
classifier_transform = None
classifier_class_names = None
classification_triggered = False
classification_confidence_threshold = 0.3  # 置信度阈值

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


# ====== 归一化参数动态加载 ======
def load_normalization_params(point_num):
    """动态加载特定点位的归一化参数

    Args:
        point_num: 点位编号 (1-10)

    Returns:
        dict: 包含STATE_MEAN, STATE_STD, ACTION_MEAN, ACTION_STD等参数的字典
    """
    # 使用各点位自己的归一化参数（与训练时一致）
    norm_file = f"/home/zeno-yanan/Zeno-hanger/normalization/point{point_num}_v30_norm.py"

    try:
        spec = importlib.util.spec_from_file_location(f"point{point_num}_norm", norm_file)
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

        rospy.loginfo(f"成功加载 point{point_num} 的归一化参数")
        return params

    except Exception as e:
        rospy.logerr(f"加载归一化参数失败 (point{point_num}): {e}")
        # 回退到point1作为默认值
        if point_num != 1:
            rospy.logwarn("回退到 point1 的归一化参数")
            return load_normalization_params(1)
        else:
            raise


# ====== 模型加载 ======
def load_policy_for_point(point_num, device_str):
    """加载特定点位的ACT策略模型

    Args:
        point_num: 点位编号 (1-10)
        device_str: 设备 ('cuda' 或 'cpu')

    Returns:
        ACTPolicy: 加载的策略模型
    """
    model_path = f"/home/zeno-yanan/Zeno-hanger/A_Exp2/exp2_point{point_num}/pretrained_model"

    try:
        local_path = Path(model_path).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        rospy.loginfo(f"正在加载 point{point_num} 模型: {model_path}")

        policy = ACTPolicy.from_pretrained(pretrained_name_or_path=str(local_path))
        policy = policy.to(device_str)
        policy.eval()

        rospy.loginfo("=" * 70)
        rospy.loginfo(f"[INFO] Point{point_num} 模型加载成功")
        rospy.loginfo(f"  chunk_size: {policy.config.chunk_size}")
        rospy.loginfo(f"  n_action_steps: {policy.config.n_action_steps}")
        rospy.loginfo("=" * 70)

        return policy

    except Exception as e:
        rospy.logerr(f"加载模型失败 (point{point_num}): {e}")
        raise


# ====== 分类器加载 ======
def load_classifier(model_path, device_str):
    """加载点位分类器模型

    Args:
        model_path: 分类器模型路径
        device_str: 设备 ('cuda' 或 'cpu')

    Returns:
        tuple: (model, transform, class_names)
    """
    try:
        checkpoint = torch.load(model_path, map_location=device_str, weights_only=False)
        class_names = checkpoint.get("class_names", [f"point{i}" for i in range(1, 11)])
        num_classes = len(class_names)

        model = PointClassifier(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device_str)
        model.eval()

        # 创建transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_acc = checkpoint.get("val_acc", None)
        epoch = checkpoint.get("epoch", None)
        rospy.loginfo(f"分类器加载成功: {model_path}")
        if epoch is not None:
            rospy.loginfo(f"  Epoch: {epoch}, Val Acc: {val_acc:.4f}" if val_acc else f"  Epoch: {epoch}")
        rospy.loginfo(f"  Classes: {class_names}")

        return model, transform, class_names

    except Exception as e:
        rospy.logerr(f"加载分类器失败: {e}")
        raise


# ====== 分类函数 ======
def classify_current_scene(image_bgr):
    """对当前场景进行分类并返回点位编号

    Args:
        image_bgr: BGR格式的numpy数组图像

    Returns:
        tuple: (point_num, confidence) 点位编号和置信度
    """
    global classifier_model, classifier_transform, classifier_class_names, device

    try:
        # 将BGR转换为RGB PIL图像
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # 预处理
        img_tensor = classifier_transform(pil_image).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            logits = classifier_model(img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        # 获取最高置信度的预测
        top_prob, top_idx = probs.max(0)
        pred_name = classifier_class_names[top_idx.item()]
        confidence = top_prob.item()

        # 提取点位编号
        point_num = int(pred_name.replace("point", ""))

        rospy.loginfo(f"分类结果: {pred_name} (置信度: {confidence:.4f})")

        return point_num, confidence

    except Exception as e:
        rospy.logerr(f"分类失败: {e}")
        return None, 0.0


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
    global success_key, classification_triggered
    try:
        if key == keyboard.Key.space:
            success_key = True
        elif hasattr(key, 'char'):
            if key.char == "c":
                # 切换遥操模式
                current_mode = rospy.get_param("/is_teleop", False)
                rospy.set_param("/is_teleop", not current_mode)
                rospy.loginfo(f"遥操模式切换: {not current_mode}")
            elif key.char == "p":
                # 触发点位分类
                classification_triggered = True
                rospy.loginfo("触发点位分类...")
    except AttributeError:
        pass


def main():
    global current_point, current_policy, current_norm_params
    global classifier_model, classifier_transform, classifier_class_names
    global classification_triggered, device, smoothed_action

    # 初始化 ROS 节点
    rospy.init_node("hanger_act_multi_model")
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
    parser = argparse.ArgumentParser(description="多模型ACT策略部署")
    parser.add_argument("--classifier-path", type=str,
                        default="/home/zeno-yanan/Zeno-hanger/A_Exp2/point_classifier/point_classifier_best.pth",
                        help="分类器模型路径")
    parser.add_argument("--default-point", type=int, default=1,
                        help="默认点位编号 (1-10)")
    parser.add_argument("--rate", type=float, default=10.0,
                        help="控制频率 Hz (default: 10)")
    parser.add_argument("--smoothing", type=float, default=0.3,
                        help="EMA平滑系数 (0=无平滑, 1=无历史)")
    parser.add_argument("--no-smoothing", action="store_true",
                        help="禁用EMA平滑")
    parser.add_argument("--action-steps", type=int, default=None,
                        help="重新推理前的动作步数")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="分类置信度阈值 (default: 0.3)")

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

    # 加载分类器
    rospy.loginfo("正在加载点位分类器...")
    classifier_model, classifier_transform, classifier_class_names = load_classifier(
        args.classifier_path, device
    )
    classification_confidence_threshold = args.confidence_threshold

    # 启动时不加载任何模型，等待按 'p' 键触发分类
    current_norm_params = None
    current_policy = None
    current_point = None
    rospy.loginfo("启动完成，等待按 'p' 键进行点位分类和模型加载...")

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
    rospy.loginfo(f"  当前点位: {'未加载' if current_point is None else f'point{current_point}'}")
    rospy.loginfo(f"  控制频率: {args.rate} Hz")
    rospy.loginfo(f"  EMA平滑: {ENABLE_SMOOTHING}, alpha={SMOOTHING_ALPHA}")
    rospy.loginfo(f"  分类置信度阈值: {classification_confidence_threshold}")
    rospy.loginfo("  按 'p' 键触发点位分类和模型加载")
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

        # 检查是否触发了分类
        if classification_triggered:
            rospy.loginfo("=" * 70)
            rospy.loginfo("[分类] 正在进行点位分类...")

            # 运行分类
            point_num, confidence = classify_current_scene(latest_imgs["wide_top"])

            if point_num is not None:
                # 检查置信度
                if confidence < classification_confidence_threshold:
                    rospy.logwarn(f"置信度过低 ({confidence:.4f} < {classification_confidence_threshold})")
                    if current_point is not None:
                        rospy.logwarn(f"保持当前模型 point{current_point}")
                    else:
                        rospy.logwarn("未加载任何模型")
                elif point_num != current_point:
                    # 加载新模型（首次加载或切换）
                    action_desc = "加载" if current_point is None else f"切换模型: point{current_point} ->"
                    rospy.loginfo(f"{action_desc} point{point_num}")
                    try:
                        # 加载新的归一化参数和模型
                        current_norm_params = load_normalization_params(point_num)
                        current_policy = load_policy_for_point(point_num, device)

                        # 覆盖 n_action_steps 配置
                        if args.action_steps is not None:
                            current_policy.config.n_action_steps = args.action_steps
                            rospy.loginfo(f"覆盖 n_action_steps 为: {args.action_steps}")

                        current_policy.reset()

                        # 重置平滑动作
                        smoothed_action["left"] = None
                        smoothed_action["right"] = None

                        current_point = point_num
                        rospy.loginfo(f"成功加载 point{point_num} 模型")
                    except Exception as e:
                        rospy.logerr(f"模型加载失败: {e}")
                        if current_point is not None:
                            rospy.logwarn(f"保持使用 point{current_point} 模型")
                        else:
                            rospy.logwarn("未加载任何模型")
                else:
                    rospy.loginfo(f"分类结果与当前点位相同 (point{current_point})，无需切换")
            else:
                rospy.logerr("分类失败")
                if current_point is not None:
                    rospy.logwarn(f"保持当前模型 point{current_point}")
                else:
                    rospy.logwarn("未加载任何模型")

            rospy.loginfo("=" * 70)
            classification_triggered = False

        # 如果没有加载模型，跳过推理
        if current_policy is None:
            Replaced = rospy.get_param("/is_teleop", False)
            master_flag_topic.publish(Bool(not Replaced))
            rate.sleep()
            continue

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
            rospy.loginfo(f"[DIAG] Step {step_count} | 当前点位: point{current_point}")
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

        rospy.loginfo_throttle(2.0, f"Actions sent - Step {step_count} | Point{current_point}")

        rate.sleep()


if __name__ == "__main__":
    main()
