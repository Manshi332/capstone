"""
preprocessing.py — Convert real hand images → skeleton renders.
"""

import os
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── MediaPipe import with friendly error ──────────────────────────────────────
try:
    import mediapipe as mp
except ImportError:
    raise ImportError(
        "MediaPipe not installed. Run:  pip install mediapipe"
    )

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_FINGER_COLORS = {
    "wrist":  (255, 255, 255),  
    "thumb":  (0,   0,   255),   
    "index":  (0,   255, 0  ),  
    "middle": (255, 165, 0  ),  
    "ring":   (255, 0,   255),  
    "pinky":  (0,   255, 255),   
}

def _landmark_color(idx: int) -> tuple:
    """Return BGR color for a given landmark index."""
    if idx == 0:
        return _FINGER_COLORS["wrist"]
    elif 1 <= idx <= 4:
        return _FINGER_COLORS["thumb"]
    elif 5 <= idx <= 8:
        return _FINGER_COLORS["index"]
    elif 9 <= idx <= 12:
        return _FINGER_COLORS["middle"]
    elif 13 <= idx <= 16:
        return _FINGER_COLORS["ring"]
    else:
        return _FINGER_COLORS["pinky"]


# ─── Core render function ─────────────────────────────────────────────────────

def render_skeleton(
    hand_landmarks,
    img_size: int = 64,
    draw_style: str = "rich",     
    connection_color=(180, 180, 180),
    joint_radius: int = 3,
    connection_thickness: int = 1,
) -> np.ndarray:
    mp_hands = mp.solutions.hands
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    lm = hand_landmarks.landmark
    H = W = img_size

    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        x1 = int(lm[start_idx].x * W)
        y1 = int(lm[start_idx].y * H)
        x2 = int(lm[end_idx].x * W)
        y2 = int(lm[end_idx].y * H)
        # Clamp to canvas
        x1, y1 = np.clip(x1, 0, W-1), np.clip(y1, 0, H-1)
        x2, y2 = np.clip(x2, 0, W-1), np.clip(y2, 0, H-1)
        cv2.line(canvas, (x1, y1), (x2, y2), connection_color, connection_thickness)

    for idx, landmark in enumerate(lm):
        x = int(np.clip(landmark.x * W, 0, W-1))
        y = int(np.clip(landmark.y * H, 0, H-1))

        if draw_style == "rich":
            color = _landmark_color(idx)
        else:
            color = (0, 255, 0)   # all green (simple mode)

        cv2.circle(canvas, (x, y), joint_radius, color, -1)   # filled circle

    return canvas


# ─── Per-image processing ─────────────────────────────────────────────────────

def process_image(
    img_path: str,
    hands_detector,
    img_size: int = 64,
    draw_style: str = "rich",
) -> np.ndarray | None:
  
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        log.warning(f"Could not read image: {img_path}")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None   
    return render_skeleton(results.multi_hand_landmarks[0], img_size=img_size, draw_style=draw_style)


# ─── Folder-level conversion ──────────────────────────────────────────────────

def convert_folder(
    input_dir: str,
    output_dir: str,
    img_size: int = 64,
    draw_style: str = "rich",
    dry_run: bool = False,
    visualize: bool = False,
):

    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No subfolders found in {input_dir}")

    log.info(f"Found {len(class_dirs)} class folders: {[d.name for d in class_dirs]}")
    log.info(f"Output dir : {output_dir}")
    log.info(f"Image size : {img_size}x{img_size}")
    log.info(f"Draw style : {draw_style}")
    log.info(f"Dry run    : {dry_run}")

    total_processed = 0
    total_skipped   = 0
    skipped_paths   = []

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.4,   
    ) as hands:

        for class_dir in class_dirs:
            letter = class_dir.name
            out_class_dir = output_dir / letter

            img_paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + \
                        list(class_dir.glob("*.jpeg"))

            if not img_paths:
                log.warning(f"  [{letter}] No images found, skipping folder.")
                continue

            if not dry_run:
                out_class_dir.mkdir(parents=True, exist_ok=True)

            class_ok   = 0
            class_skip = 0

            for img_path in tqdm(img_paths, desc=f"  [{letter}]", unit="img", leave=False):
                skeleton = process_image(
                    str(img_path), hands, img_size=img_size, draw_style=draw_style
                )

                if skeleton is None:
                    class_skip   += 1
                    total_skipped += 1
                    skipped_paths.append(str(img_path))
                    continue

                class_ok       += 1
                total_processed += 1

                if not dry_run:
                    out_path = out_class_dir / img_path.name
                    cv2.imwrite(str(out_path), skeleton)

                    # Optionally show side-by-side preview for first image per class
                    if visualize and class_ok == 1:
                        orig = cv2.imread(str(img_path))
                        orig_resized = cv2.resize(orig, (img_size * 4, img_size * 4))
                        skel_large   = cv2.resize(skeleton, (img_size * 4, img_size * 4))
                        preview = np.hstack([orig_resized, skel_large])
                        cv2.imshow(f"Preview: {letter}  (original | skeleton)", preview)
                        cv2.waitKey(500)

            log.info(f"  [{letter}]  saved: {class_ok:4d}   skipped: {class_skip:3d}")

    # Save skipped log
    if not dry_run and skipped_paths:
        skip_log = output_dir / "skipped_log.txt"
        with open(skip_log, "w") as f:
            f.write("\n".join(skipped_paths))
        log.info(f"\nSkipped log saved to: {skip_log}")

    if visualize:
        cv2.destroyAllWindows()

    # ── Summary ────────────────────────────────────────────────────────────────
    total = total_processed + total_skipped
    skip_pct = (total_skipped / total * 100) if total > 0 else 0
    print("\n" + "═" * 50)
    print(f"  Total images   : {total}")
    print(f"  Saved          : {total_processed}")
    print(f"  Skipped (no hand detected): {total_skipped}  ({skip_pct:.1f}%)")
    if not dry_run:
        print(f"  Output folder  : {output_dir}")
    print("═" * 50)

    if skip_pct > 30:
        log.warning(
            f"Over 30% of images were skipped! "
            "Consider lowering --min_detection_confidence or check your input images."
        )

    return total_processed, total_skipped


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert real ASL hand images → skeleton renders (MediaPipe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full conversion with preview:
  python preprocess_to_skeleton.py \\
      --input_dir data/extra_real --output_dir data/extra_skeleton --vis

  # Simple style (all-green joints):
  python preprocess_to_skeleton.py \\
      --input_dir data/extra_real --output_dir data/extra_skeleton --draw_style simple

  # Dry run only (no files written):
  python preprocess_to_skeleton.py --input_dir data/extra_real --dry_run
        """
    )
    parser.add_argument("--input_dir",  required=True,
                        help="Root folder with A/, B/, ... subfolders of real hand images")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save skeleton renders (default: input_dir + '_skeleton')")
    parser.add_argument("--img_size",   type=int, default=64,
                        help="Output image size in pixels (default: 64, must match IMG_SIZE in dataset.py)")
    parser.add_argument("--draw_style", choices=["rich", "simple"], default="rich",
                        help="'rich'=colored joints per finger (matches custom dataset), 'simple'=all-green")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Only count/report — do not write any files")
    parser.add_argument("--vis",        action="store_true",
                        help="Show side-by-side preview for first image of each class")

    args = parser.parse_args()

    output_dir = args.output_dir or (args.input_dir.rstrip("/\\") + "_skeleton")

    convert_folder(
        input_dir  = args.input_dir,
        output_dir = output_dir,
        img_size   = args.img_size,
        draw_style = args.draw_style,
        dry_run    = args.dry_run,
        visualize  = args.vis,
    )