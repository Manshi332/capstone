"""
predict.py — Single image ASL prediction with confidence scores.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from dataset import get_val_transforms, LABEL_TO_LETTER, NUM_CLASSES
from custom_cnn import build_custom_cnn
from transfer_model import build_transfer_model

try:
    import mediapipe as mp
    _mp_hands  = mp.solutions.hands
    MEDIAPIPE_OK = True
except AttributeError:
    MEDIAPIPE_OK = False

_FINGER_COLORS = {
    0:       (255, 255, 255), 
    "thumb": (0,   0,   255),  
    "index": (0,   255, 0  ), 
    "mid":   (255, 165, 0  ),  
    "ring":  (255, 0,   255),  
    "pinky": (0,   255, 255),  
}

def _joint_color(idx):
    if idx == 0: return _FINGER_COLORS[0]
    elif idx <= 4:  return _FINGER_COLORS["thumb"]
    elif idx <= 8:  return _FINGER_COLORS["index"]
    elif idx <= 12: return _FINGER_COLORS["mid"]
    elif idx <= 16: return _FINGER_COLORS["ring"]
    else:           return _FINGER_COLORS["pinky"]


def render_skeleton_from_landmarks(hand_landmarks, img_size=64):
    """Render colored skeleton on black background — matches training domain."""
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    lm = hand_landmarks.landmark
    W = H = img_size

    for conn in _mp_hands.HAND_CONNECTIONS:
        s, e = conn
        x1 = int(np.clip(lm[s].x * W, 0, W-1))
        y1 = int(np.clip(lm[s].y * H, 0, H-1))
        x2 = int(np.clip(lm[e].x * W, 0, W-1))
        y2 = int(np.clip(lm[e].y * H, 0, H-1))
        cv2.line(canvas, (x1, y1), (x2, y2), (180, 180, 180), 1)

    # Draw joints on top
    for i, point in enumerate(lm):
        x = int(np.clip(point.x * W, 0, W-1))
        y = int(np.clip(point.y * H, 0, H-1))
        cv2.circle(canvas, (x, y), 3, _joint_color(i), -1)

    return canvas  # BGR numpy array


def detect_and_render(image_path: str):
    """
    Load image, detect hand with MediaPipe, render skeleton.
    Returns (skeleton_pil_image, raw_crop_pil, detected).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if not MEDIAPIPE_OK:
        print("  [warn] mediapipe not available — cannot render skeleton")
        print("         Fix: pip install mediapipe==0.10.14")
        # Fallback: use raw image (predictions will be less accurate)
        pil = Image.fromarray(cv2.resize(img_rgb, (64, 64)))
        return pil, pil, False

    with _mp_hands.Hands(
        static_image_mode=True, max_num_hands=1,
        model_complexity=1, min_detection_confidence=0.3,
    ) as hands:
        results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        print("  [warn] No hand detected — using raw image (accuracy will be lower)")
        pil = Image.fromarray(cv2.resize(img_rgb, (64, 64)))
        return pil, pil, False

    skeleton_bgr = render_skeleton_from_landmarks(results.multi_hand_landmarks[0])
    skeleton_rgb = cv2.cvtColor(skeleton_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    lm = results.multi_hand_landmarks[0]
    xs = [p.x * w for p in lm.landmark]
    ys = [p.y * h for p in lm.landmark]
    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    half   = max(int((max(xs)-min(xs))/2), int((max(ys)-min(ys))/2)) + 40
    x1,y1  = max(0,cx-half), max(0,cy-half)
    x2,y2  = min(w,cx+half), min(h,cy+half)
    raw_crop = Image.fromarray(img_rgb[y1:y2, x1:x2])

    print(f"  [info] Hand detected — skeleton rendered at 64x64")
    return Image.fromarray(skeleton_rgb), raw_crop, True


# ─── Model Loader ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, model_type: str, backbone: str = "mobilenetv2"):
    if model_type == "cnn":
        model = build_custom_cnn(num_classes=NUM_CLASSES)
    else:
        model = build_transfer_model(backbone=backbone, num_classes=NUM_CLASSES, freeze_backbone=False)
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


# ─── Predict ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, device) -> dict:
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    probs  = F.softmax(logits, dim=1).squeeze()

    confidence, pred_class = probs.max(dim=0)
    top5_probs, top5_idx   = torch.topk(probs, k=5)
    top5 = [
        {"letter": LABEL_TO_LETTER[c.item()], "confidence": round(p.item() * 100, 2)}
        for c, p in zip(top5_idx, top5_probs)
    ]
    return {
        "predicted_letter": LABEL_TO_LETTER[pred_class.item()],
        "confidence":        round(confidence.item() * 100, 2),
        "top5":              top5,
    }


# ─── Visualization ───────────────────────────────────────────────────────────

def visualize_prediction(image_path: str, skeleton_img, raw_crop, result: dict, save_path=None):
    original = Image.open(image_path).convert("RGB")
    top5     = result["top5"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4),
                             gridspec_kw={"width_ratios": [2, 1, 1, 2]})

    axes[0].imshow(original);     axes[0].axis("off"); axes[0].set_title("Input photo")
    axes[1].imshow(raw_crop);     axes[1].axis("off"); axes[1].set_title("Hand crop")
    axes[2].imshow(skeleton_img); axes[2].axis("off"); axes[2].set_title("Skeleton (model input)")

    letters = [t["letter"] for t in top5]
    confs   = [t["confidence"] for t in top5]
    colors  = ["#2196F3" if i == 0 else "#90CAF9" for i in range(5)]
    bars    = axes[3].barh(letters[::-1], confs[::-1], color=colors[::-1], height=0.55)
    axes[3].set_xlim(0, 105)
    axes[3].set_xlabel("Confidence (%)")
    axes[3].set_title(f"Predicted: {result['predicted_letter']}  ({result['confidence']:.1f}%)",
                      fontweight="bold")
    for bar, c in zip(bars, confs[::-1]):
        axes[3].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f"{c:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization → {save_path}")
    else:
        plt.show()
    plt.close()


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict ASL letter from a single image")
    parser.add_argument("--image",      required=True)
    parser.add_argument("--model",      required=True)
    parser.add_argument("--model_type", choices=["cnn", "transfer"], required=True)
    parser.add_argument("--backbone",   default="mobilenetv2")
    parser.add_argument("--save_plot",  default=None)
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: image not found at {args.image}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.model, args.model_type, args.backbone).to(device)

    skeleton_pil, raw_crop, detected = detect_and_render(args.image)

    transform = get_val_transforms()
    tensor    = transform(skeleton_pil).unsqueeze(0)
    result    = predict(model, tensor, device)

    print(f"\n{'─'*38}")
    print(f"  Image      : {args.image}")
    print(f"  Model      : {args.model_type}")
    print(f"  Skeleton   : {'rendered' if detected else 'fallback (no hand detected)'}")
    print(f"{'─'*38}")
    print(f"  Predicted  : {result['predicted_letter']}")
    print(f"  Confidence : {result['confidence']:.2f}%")
    print(f"\n  Top-5:")
    for t in result["top5"]:
        bar = "█" * int(t["confidence"] / 5)
        print(f"    {t['letter']:>2}  {bar:<20}  {t['confidence']:5.1f}%")

    visualize_prediction(args.image, skeleton_pil, raw_crop, result, save_path=args.save_plot)