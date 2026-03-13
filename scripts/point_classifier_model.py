"""
Point Classifier Model
基于 ResNet-18 的毛巾起始点位分类器 (10 类)
输入: wide_top 相机单帧图像
输出: point1 - point10 的分类预测
"""

import torch
import torch.nn as nn
from torchvision import models


class PointClassifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = PointClassifier(num_classes=10)
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # [1, 10]
