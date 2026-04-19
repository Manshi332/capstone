"""
collect_data.py — Collect custom ASL dataset using webcam + MediaPipe.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

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

VALID_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ROI_SIZE         = 128
ROI_PADDING      = 45
CAPTURE_INTERVAL = 0.15   # seconds between auto-saves


# ─── ROI Extraction ───────────────────────────────────────────────────────────

def extract_roi(frame, landmarks, padding=ROI_PADDING, size=ROI_SIZE):
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in landmarks.landmark]
    ys = [lm.y * h for lm in landmarks.landmark]

    cx   = int(np.mean(xs))
    cy   = int(np.mean(ys))
    half = max(int((max(xs) - min(xs)) / 2), int((max(ys) - min(ys)) / 2)) + padding

    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)

    if x2 <= x1 or y2 <= y1:
        return None, None

    roi = cv2.resize(frame[y1:y2, x1:x2], (size, size))
    return roi, (x1, y1, x2, y2)


# ─── Overlay ─────────────────────────────────────────────────────────────────

def draw_ui(frame, letter, counts, target, bbox):
    h, w = frame.shape[:2]

    # Status bar background
    cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)

    if letter:
        n      = counts.get(letter, 0)
        pct    = int((n / target) * (w - 20))
        status = f"Capturing: {letter}   [ {n} / {target} ]"
        color  = (50, 220, 50)
        cv2.rectangle(frame, (10, 48), (10 + pct, 57), color, -1)
        cv2.rectangle(frame, (10, 48), (w - 10, 57), (70, 70, 70), 1)
    else:
        status = "Press a letter key  (A-Z)  to start capturing"
        color  = (180, 180, 180)

    cv2.putText(frame, status, (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)

    # Bounding box
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

    # Bottom bar — class progress
    done_count = sum(1 for l in VALID_LETTERS if counts.get(l, 0) >= target)
    summary = "  ".join(
        f"{l}:{'OK' if counts.get(l,0)>=target else counts.get(l,0)}"
        for l in VALID_LETTERS if counts.get(l, 0) > 0
    )
    cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, summary[:100], (8, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)
    cv2.putText(frame, f"[ESC]quit [r]reset current class   Done: {done_count}/26",
                (8, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 90), 1)

    return frame


# ─── Main ────────────────────────────────────────────────────────────────────

def collect_data(output_dir="data/custom", target=150, camera_id=0):

    if not MEDIAPIPE_OK:
        print("\n" + "=" * 55)
        print("  ERROR: mediapipe.solutions not found.")
        print("  Fix: pip install mediapipe==0.10.14")
        print("=" * 55)
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    counts = {}
    for l in VALID_LETTERS:
        (out / l).mkdir(exist_ok=True)
        counts[l] = len(list((out / l).glob("*.jpg")))

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_id}")
        print("Try --camera 1 if you have multiple cameras.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nASL Data Collector — press a letter key to start capturing.")
    print("See the docstring at the top of this file for full instructions.\n")

    active_letter     = None
    last_capture_time = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read error.")
                break

            frame   = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            bbox = None

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                roi, bbox = extract_roi(frame, lm)
                if roi is not None:
                    # Inset ROI preview (top-right corner)
                    preview = cv2.resize(roi, (100, 100))
                    frame[62:162, -110:-10] = preview
                    cv2.rectangle(frame, (frame.shape[1]-112, 60),
                                  (frame.shape[1]-8, 164), (0, 200, 255), 1)

                    # Auto-capture
                    now = time.time()
                    if (active_letter and
                            now - last_capture_time >= CAPTURE_INTERVAL and
                            counts[active_letter] < target):

                        n = counts[active_letter]
                        cv2.imwrite(str(out / active_letter / f"{n:06d}.jpg"), roi)
                        counts[active_letter] += 1
                        last_capture_time = now

                        if counts[active_letter] >= target:
                            print(f"  ✓  '{active_letter}' complete — {target} images saved.")
                            active_letter = None

            frame = draw_ui(frame, active_letter, counts, target, bbox)
            cv2.imshow("ASL Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            # ── Check letter keys FIRST (so Q captures the letter, not quits) ──
            if key != 255:
                ch = chr(key).upper()

                if ch == "R" and active_letter:
                    # Reset current class
                    for p in (out / active_letter).glob("*.jpg"):
                        p.unlink()
                    counts[active_letter] = 0
                    print(f"  Reset '{active_letter}' — recapturing from 0.")

                elif ch in VALID_LETTERS:
                    # Normal letter capture trigger (includes Q)
                    if counts[ch] >= target:
                        print(f"  '{ch}' already done ({target} images). Press [r] to reset.")
                    else:
                        active_letter = ch
                        print(f"  → Capturing '{ch}' — existing: {counts[ch]}")

                

            # ESC always quits immediately
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    print("\n" + "─" * 45)
    total = 0
    for l in VALID_LETTERS:
        n   = counts.get(l, 0)
        bar = "█" * min(n // 5, 30)
        tag = "✓" if n >= target else str(n)
        print(f"  {l}: {bar:<30} {tag}")
        total += n
    print(f"\n  Total: {total} images  →  {out.resolve()}")
   
# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect ASL images via webcam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_dir",       default="data/custom")
    parser.add_argument("--target_per_class", type=int, default=150)
    parser.add_argument("--camera",           type=int, default=0)
    args = parser.parse_args()

    collect_data(args.output_dir, args.target_per_class, args.camera)
