"""
webcam.py — Real-time ASL recognition via webcam.
"""

import argparse
import collections
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import get_val_transforms, LABEL_TO_LETTER, NUM_CLASSES
from custom_cnn import build_custom_cnn
from transfer_model import build_transfer_model
from predict import render_skeleton_from_landmarks

# ─── MediaPipe import ────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
    MEDIAPIPE_OK = True
except AttributeError:
    MEDIAPIPE_OK = False


# ─── Constants ───────────────────────────────────────────────────────────────

BUFFER_SIZE    = 10
CONF_THRESHOLD = 0.60
ROI_PADDING    = 40


# ─── Model Loader ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, model_type: str, backbone: str = "mobilenetv2"):
    """
    Build the correct architecture then load weights from checkpoint.

    For model_type="cnn":      backbone is ignored, CustomCNN is built.
    For model_type="transfer": backbone must match the .pth file exactly.
      - resnet50.pth     → backbone="resnet50"
      - vgg161.pth       → backbone="vgg16"
      - mobilenetv2.pth  → backbone="mobilenetv2"
    """
    if model_type == "cnn":
        model = build_custom_cnn(num_classes=NUM_CLASSES)
        print(f"Loaded: Custom CNN  ←  {checkpoint_path}")
    else:
        model = build_transfer_model(
            backbone=backbone, num_classes=NUM_CLASSES, freeze_backbone=False
        )
        print(f"Loaded: {backbone}  ←  {checkpoint_path}")

    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


# ─── Inference ───────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(model, hand_landmarks, transform, device):
    """
    Render skeleton from MediaPipe landmarks, then run inference.
    This matches the training domain exactly (skeleton images, not raw photos).
    """
    skeleton_bgr = render_skeleton_from_landmarks(hand_landmarks, img_size=64)
    skeleton_rgb = cv2.cvtColor(skeleton_bgr, cv2.COLOR_BGR2RGB)
    pil    = Image.fromarray(skeleton_rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    probs  = torch.softmax(model(tensor), dim=1).squeeze()
    conf, idx = probs.max(dim=0)
    return LABEL_TO_LETTER[idx.item()], conf.item(), probs.cpu().numpy()


# ─── Bounding box (visual only) ───────────────────────────────────────────────

def get_bbox(frame, hand_landmarks, padding=ROI_PADDING):
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x1 = max(0, int(min(xs)) - padding)
    y1 = max(0, int(min(ys)) - padding)
    x2 = min(w, int(max(xs)) + padding)
    y2 = min(h, int(max(ys)) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


# ─── Overlay ─────────────────────────────────────────────────────────────────

def draw_overlay(frame, letter, conf, word, bbox, fps, model_name):
    h, w = frame.shape[:2]
    color = (50, 210, 50) if conf >= CONF_THRESHOLD else (100, 100, 210)

    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        badge = f"{letter}  {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.rectangle(frame, (x1, y1-th-12), (x1+tw+10, y1), color, -1)
        cv2.putText(frame, badge, (x1+5, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    cv2.rectangle(frame, (0, 0), (w, 48), (20, 20, 20), -1)
    cv2.putText(frame, f"ASL  |  {model_name}  |  FPS {fps:.0f}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

    cv2.rectangle(frame, (0, h-52), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, f"Word: {word if word else '...'}", (10, h-22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (100, 220, 255), 2)
    cv2.putText(frame, "[s]append  [c]clear  [SPACE]space  [w]save  [ESC]quit",
                (10, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 110, 110), 1)
    return frame


# ─── Main Loop ───────────────────────────────────────────────────────────────

def run_webcam(model, device, model_name="Model", camera_id=0, save_dir="results"):

    if not MEDIAPIPE_OK:
        print("ERROR: mediapipe.solutions not found.")
        print("Fix: pip install mediapipe==0.10.14")
        return

    transform = get_val_transforms()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    buf     = collections.deque(maxlen=BUFFER_SIZE)
    word    = ""
    fps_buf = collections.deque(maxlen=30)
    prev_t  = time.time()

    print(f"\nWebcam started — {model_name}")
    print("Controls: [s] append letter  [c] clear  [SPACE] space  [w] save  [ESC] quit\n")

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            fps_buf.append(1.0 / max(now - prev_t, 1e-6))
            prev_t = now
            fps    = np.mean(fps_buf)

            frame   = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            letter, conf, bbox = "?", 0.0, None

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                bbox = get_bbox(frame, lm)
                # Pass landmarks to infer — skeleton is rendered inside
                letter, conf, _ = infer(model, lm, transform, device)
                if conf >= CONF_THRESHOLD:
                    buf.append(letter)

            smoothed = collections.Counter(buf).most_common(1)[0][0] if buf else "?"
            frame    = draw_overlay(frame, smoothed, conf, word, bbox, fps, model_name)
            cv2.imshow(f"ASL — {model_name}", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord("c"):
                word = ""; buf.clear()
            elif key == ord(" "):
                word += " "
            elif key == ord("s") and smoothed != "?":
                word += smoothed; buf.clear()
                print(f"  Appended '{smoothed}' → '{word}'")
            elif key == ord("w"):
                p = Path(save_dir) / "recognized_words.txt"
                with open(p, "a") as f:
                    f.write(word.strip() + "\n")
                print(f"  Saved '{word.strip()}' → {p}")
                word = ""

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time ASL recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/webcam.py --model models/custom_cnn.pth  --model_type cnn
  python src/webcam.py --model models/resnet50.pth    --model_type transfer --backbone resnet50
  python src/webcam.py --model models/vgg161.pth      --model_type transfer --backbone vgg16
  python src/webcam.py --model models/mobilenetv2.pth --model_type transfer --backbone mobilenetv2
        """
    )
    parser.add_argument("--model",      required=True)
    parser.add_argument("--model_type", choices=["cnn", "transfer"], required=True)
    parser.add_argument("--backbone",   default="mobilenetv2",
                        choices=["mobilenetv2", "resnet50", "efficientnet", "vgg16"],
                        help="Must match the architecture used to save the .pth file")
    parser.add_argument("--camera",     type=int, default=0)
    parser.add_argument("--save_dir",   default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.model, args.model_type, args.backbone).to(device)
    name   = "Custom CNN" if args.model_type == "cnn" else args.backbone

    run_webcam(model, device, model_name=name,
               camera_id=args.camera, save_dir=args.save_dir)