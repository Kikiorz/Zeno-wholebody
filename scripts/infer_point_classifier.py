"""
Point Classifier Inference Script
点位分类器推理脚本 - 输入图片，输出 point1-point10 分类结果

用法:
  单张图片:  python3 infer_point_classifier.py --image /path/to/image.jpg
  批量目录:  python3 infer_point_classifier.py --image_dir /path/to/images/
  指定模型:  python3 infer_point_classifier.py --image img.jpg --model_path /path/to/model.pth
"""

import argparse
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

from point_classifier_model import PointClassifier


DEFAULT_MODEL_PATH = "/workspace/Mult-skill ACT/models/point_classifier/point_classifier_best.pth"


def load_model(model_path, device):
    """加载分类器模型和类别名称"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", [f"point{i}" for i in range(1, 11)])
    num_classes = len(class_names)

    model = PointClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_acc = checkpoint.get("val_acc", None)
    epoch = checkpoint.get("epoch", None)
    print(f"Loaded model from: {model_path}")
    if epoch is not None:
        print(f"  Epoch: {epoch}, Val Acc: {val_acc:.4f}" if val_acc else f"  Epoch: {epoch}")
    print(f"  Classes: {class_names}")

    return model, class_names


def get_transform():
    """与训练时验证集一致的预处理"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def infer_single(model, image_path, transform, class_names, device, top_k=3):
    """对单张图片进行推理"""
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_k = min(top_k, len(class_names))
    top_probs, top_indices = probs.topk(top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append((class_names[idx.item()], prob.item()))

    return results


def main():
    parser = argparse.ArgumentParser(description="Point classifier inference")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Path to a directory of images")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Show top-k predictions (default: 3)")
    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("Please provide --image or --image_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, class_names = load_model(args.model_path, device)
    transform = get_transform()

    # 收集图片列表
    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    if args.image_dir:
        img_dir = Path(args.image_dir)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_paths.extend(sorted(img_dir.glob(ext)))

    if not image_paths:
        print("No images found.")
        return

    print(f"\nProcessing {len(image_paths)} image(s)...\n")
    print("-" * 60)

    for img_path in image_paths:
        results = infer_single(model, img_path, transform, class_names, device, args.top_k)
        pred_name, pred_conf = results[0]
        print(f"{img_path.name}")
        print(f"  Prediction: {pred_name}  (confidence: {pred_conf:.4f})")
        if len(results) > 1:
            top_str = ", ".join([f"{name}: {conf:.4f}" for name, conf in results])
            print(f"  Top-{len(results)}: {top_str}")
        print()


if __name__ == "__main__":
    main()
