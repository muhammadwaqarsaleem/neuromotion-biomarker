"""
NeuroMotion - Environment Verification Script
=============================================
Validates core dependencies, versions, and local hardware video capture capabilities.
"""

import sys

def verify_libraries():
    print("=" * 60)
    print("NEUROMOTION ENVIRONMENT VERIFICATION")
    print("=" * 60)
    print(f"Python Runtime: {sys.version.split()[0]} ({sys.executable})\n")

    modules_to_test = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("streamlit", "Streamlit"),
        ("plotly", "Plotly"),
    ]

    all_passed = True
    print("Library Import & Version Check:")
    print("-" * 60)

    for mod_name, label in modules_to_test:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Available")
            print(f"  [PASS] {label:12} : v{ver}")
        except Exception as e:
            all_passed = False
            print(f"  [FAIL] {label:12} : ERROR -> {e}")

    return all_passed


def probe_camera():
    print("\n" + "-" * 60)
    print("Hardware Video Capture Probe (Camera 0):")
    print("-" * 60)

    try:
        import cv2
        # Non-blocking probe of camera index 0
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(0)
        
        if not cap.isOpened():
            # Try default backend if DSHOW did not open
            cap.release()
            cap = cv2.VideoCapture(0)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w, c = frame.shape
                print(f"  [PASS] Camera 0 accessible. Frame acquired: {w}x{h} ({c} channels)")
            else:
                print("  [WARN] Camera 0 opened, but unable to grab test frame (may be in use or virtual).")
            cap.release()
            return True
        else:
            print("  [INFO] No onboard camera detected at index 0 (headless/virtual environment or disabled).")
            return False
    except Exception as e:
        print(f"  [WARN] Camera probe raised exception: {e}")
        return False


if __name__ == "__main__":
    libs_ok = verify_libraries()
    probe_camera()
    print("\n" + "=" * 60)
    if libs_ok:
        print("STATUS: Environment successfully verified!")
    else:
        print("STATUS: Verification failed for one or more packages.")
    print("=" * 60)
    sys.exit(0 if libs_ok else 1)
