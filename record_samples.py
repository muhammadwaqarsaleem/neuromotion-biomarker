"""
NeuroMotion - Automated Sample Video Recording Utility
======================================================
CLI helper to record standardized 10-second (300 frames @ 30 FPS) clinical motor
assessment videos and save them directly to `assets/` for offline demonstration.

Usage:
    python record_samples.py
"""

import os
import sys
import time
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def get_camera_capture(camera_id: int = 0) -> cv2.VideoCapture:
    """Initialize camera with DirectShow on Windows for optimal 30 FPS capture."""
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_id)
    else:
        cap = cv2.VideoCapture(camera_id)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30.0)

    return cap


def record_trial(
    filename: str,
    title: str,
    instructions: str,
    duration_sec: float = 10.0,
    camera_id: int = 0,
) -> bool:
    """Record a 10-second standardized assessment clip with pre-roll countdown."""
    output_path = os.path.join(ASSETS_DIR, filename)

    print("\n" + "=" * 65)
    print(f"🎬 PREPARING TO RECORD: {title}")
    print("=" * 65)
    print(f"Target Output : {output_path}")
    print(f"Instructions  : {instructions}")
    print("-" * 65)

    cap = get_camera_capture(camera_id)
    if not cap.isOpened():
        print(f"❌ ERROR: Cannot access camera (index {camera_id}). Please check USB/webcam connection.")
        return False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    if not out.isOpened():
        print(f"❌ ERROR: Could not open VideoWriter for {output_path}")
        cap.release()
        return False

    # 1. PRE-ROLL COUNTDOWN (3 seconds)
    print("Position your hand in front of the camera. Starting 3-second countdown...")
    pre_roll_start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - pre_roll_start
        rem = 3.5 - elapsed
        if rem <= 0:
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Show mirror view for natural user positioning
        preview = cv2.flip(frame, 1)
        h, w = preview.shape[:2]

        # Draw countdown banner
        count_num = int(np.ceil(rem))
        cv2.putText(preview, f"RECORDING STARTS IN: {count_num}", (w // 2 - 210, h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, title, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, instructions[:55], (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 220, 240), 1, cv2.LINE_AA)

        cv2.imshow("NeuroMotion Recorder (Press ESC to cancel)", preview)
        if cv2.waitKey(1) & 0xFF == 27:
            print("⚠️ Recording cancelled by user.")
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            return False

    # 2. ACTIVE RECORDING (10.0 seconds)
    print("🔴 RECORDING IN PROGRESS! Perform the movement now...")
    start_time = time.perf_counter()
    total_frames = int(duration_sec * fps)
    frame_count = 0

    while frame_count < total_frames:
        curr_time = time.perf_counter()
        elapsed = curr_time - start_time
        if elapsed >= duration_sec:
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # In standard mode, save un-mirrored raw capture to file so downstream
        # pipelines apply mirror uniformly, or save mirrored if preferred.
        # We save frame at 640x480.
        frame_resized = cv2.resize(frame, (640, 480))
        out.write(frame_resized)
        frame_count += 1

        # Live feedback display
        display_frame = cv2.flip(frame_resized, 1)
        rem_sec = max(0.0, duration_sec - elapsed)
        prog_ratio = min(1.0, elapsed / duration_sec)

        # Recording badge & timer
        cv2.circle(display_frame, (30, 30), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(display_frame, f"REC: {rem_sec:4.1f}s remaining", (48, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"Frames: {frame_count}/{total_frames}", (48, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1, cv2.LINE_AA)

        # Progress bar
        bar_w = 400
        bar_x = (640 - bar_w) // 2
        bar_y = 440
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (30, 30, 40), -1)
        fill_w = int(prog_ratio * bar_w)
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 14), (0, 220, 100), -1)
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (100, 120, 140), 1)

        cv2.imshow("NeuroMotion Recorder (Press ESC to cancel)", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            print("⚠️ Recording cancelled.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
        print(f"✅ SUCCESS! Recorded {frame_count} frames to: {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
        return True
    else:
        print("❌ Recording failed or file empty.")
        return False


def main():
    print("\n" + "=" * 65)
    print("🎥 NEUROMOTION AI CLINICAL SAMPLE VIDEO RECORDER")
    print("=" * 65)
    print("Select which sample video you would like to record:")
    print("  [1] Normal Healthy Finger Tapping (Task 3.4 - Rapid, Wide, Steady)")
    print("  [2] Parkinsonian Bradykinesia Tapping (Task 3.4 - Progressive Decrement & Fatigue)")
    print("  [3] Rest / Postural Tremor (Task 3.15 - 4-6 Hz Rhythmic Tremor Oscillation)")
    print("  [4] Record All 3 Sequentially")
    print("  [q] Quit")
    print("-" * 65)

    choice = input("Enter choice (1/2/3/4/q): ").strip().lower()

    if choice == "1":
        record_trial(
            filename="finger_tap_normal.mp4",
            title="Healthy Normal Finger Tapping (Task 3.4)",
            instructions="Tap thumb and index finger as fast and wide as possible with regular rhythm.",
        )
    elif choice == "2":
        record_trial(
            filename="finger_tap_bradykinesia.mp4",
            title="Bradykinesia with Decrement (Task 3.4)",
            instructions="Start tapping normally, then gradually reduce aperture width and slow down.",
        )
    elif choice == "3":
        record_trial(
            filename="tremor_resting.mp4",
            title="Rest / Postural Tremor (Task 3.15)",
            instructions="Hold hand outstretched or supported, oscillating thumb/fingers at 4-6 Hz.",
        )
    elif choice == "4":
        print("\n--- RECORDING SEQUENCE 1 of 3 ---")
        record_trial(
            filename="finger_tap_normal.mp4",
            title="Healthy Normal Finger Tapping (Task 3.4)",
            instructions="Tap thumb and index finger as fast and wide as possible with regular rhythm.",
        )
        time.sleep(1.5)
        print("\n--- RECORDING SEQUENCE 2 of 3 ---")
        record_trial(
            filename="finger_tap_bradykinesia.mp4",
            title="Bradykinesia with Decrement (Task 3.4)",
            instructions="Start tapping normally, then gradually reduce aperture width and slow down.",
        )
        time.sleep(1.5)
        print("\n--- RECORDING SEQUENCE 3 of 3 ---")
        record_trial(
            filename="tremor_resting.mp4",
            title="Rest / Postural Tremor (Task 3.15)",
            instructions="Hold hand outstretched or supported, oscillating thumb/fingers at 4-6 Hz.",
        )
    else:
        print("Exiting recorder.")


if __name__ == "__main__":
    main()
