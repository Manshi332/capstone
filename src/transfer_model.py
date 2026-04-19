"""
transfer_model.py — Transfer learning models for ASL recognition.
"""

import torch
import torch.nn as nn
from torchvision import models


# ─── MobileNetV2 ─────────────────────────────────────────────────────────────

def build_mobilenetv2(num_classes: int = 26, freeze_backbone: bool = True) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


def unfreeze_mobilenetv2(model: nn.Module, n_layers: int = 5) -> nn.Module:
    layers = list(model.features.children())
    for layer in layers[-n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    return model


# ─── ResNet50 ─────────────────────────────────────────────────────────────────

def build_resnet50(num_classes: int = 26, freeze_backbone: bool = True) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def unfreeze_resnet50(model: nn.Module, layers=("layer4", "layer3")) -> nn.Module:
    for name, param in model.named_parameters():
        if any(name.startswith(l) for l in layers):
            param.requires_grad = True
    return model


# ─── EfficientNetB0 ───────────────────────────────────────────────────────────

def build_efficientnet(num_classes: int = 26, freeze_backbone: bool = True) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


# ─── VGG16 ───────────────────────────────────────────────────────────────────

def build_vgg16(num_classes: int = 26, freeze_backbone: bool = True) -> nn.Module:
    """
    VGG16 — heavy (138M params) but strong feature extractor.
    Warning: slow on CPU. Recommended for Colab GPU only.
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(25088, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def unfreeze_vgg16(model: nn.Module, n_layers: int = 4) -> nn.Module:
    children = list(model.features.children())
    for layer in children[-n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    return model


# ─── Factory ─────────────────────────────────────────────────────────────────

SUPPORTED_BACKBONES = {
    "mobilenetv2":  build_mobilenetv2,
    "resnet50":     build_resnet50,
    "efficientnet": build_efficientnet,
    "vgg16":        build_vgg16,
}


def build_transfer_model(
    backbone: str = "mobilenetv2",
    num_classes: int = 26,
    freeze_backbone: bool = True,
) -> nn.Module:
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose: {list(SUPPORTED_BACKBONES)}")

    model     = SUPPORTED_BACKBONES[backbone](num_classes=num_classes, freeze_backbone=freeze_backbone)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Backbone  : {backbone}")
    print(f"Params    : {total:,}  (trainable: {trainable:,} = {100*trainable/total:.1f}%)")
    return model


def unfreeze_top_layers(model: nn.Module, backbone: str, n: int = 5) -> nn.Module:
    if backbone == "mobilenetv2":
        return unfreeze_mobilenetv2(model, n_layers=n)
    elif backbone == "resnet50":
        return unfreeze_resnet50(model)
    elif backbone == "efficientnet":
        for param in list(model.features.parameters())[-n * 10:]:
            param.requires_grad = True
        return model
    elif backbone == "vgg16":
        return unfreeze_vgg16(model, n_layers=n)
    return model


# ─── Sanity check ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    x = torch.randn(2, 3, 64, 64)
    for backbone in SUPPORTED_BACKBONES:
        model = build_transfer_model(backbone=backbone, num_classes=26)
        out   = model(x)
        print(f"  {backbone}: output {out.shape}\n")