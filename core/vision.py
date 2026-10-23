"""
Vision Pipeline & Kinematic Tracking
====================================
Core computer vision pipeline for NeuroMotion. Provides real-time hand tracking,
landmark extraction, and thread-safe rolling kinematic buffering for digital biomarker
analysis (tremor frequency, finger tapping, bradykinesia estimation).
"""

import threading
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np

# Landmark Indices for Clinical Motor Assessment
WRIST_IDX: int = 0
THUMB_TIP_IDX: int = 4
INDEX_TIP_IDX: int = 8
MIDDLE_MCP_IDX: int = 9
MIDDLE_TIP_IDX: int = 12

KEY_LANDMARK_INDICES: Tuple[int, ...] = (
    WRIST_IDX,
    THUMB_TIP_IDX,
    INDEX_TIP_IDX,
    MIDDLE_MCP_IDX,
    MIDDLE_TIP_IDX,
)


class HandednessInfo(dict):
    """
    Structured container for hand classification (label and confidence score).
    
    Supports dictionary access (d["label"], d["score"]), attribute access
    (d.label, d.score), equality check with string ("Left" / "Right"),
    and human-readable string representation.
    """

    def __init__(self, label: str, score: float):
        super().__init__(label=label, score=score)
        self.label: str = label
        self.score: float = float(score)

    def __str__(self) -> str:
        return f"{self.label} ({self.score:.2f})"

    def __repr__(self) -> str:
        return f"HandednessInfo(label='{self.label}', score={self.score:.4f})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.label == other
        return super().__eq__(other)


class HandTracker:
    """
    MediaPipe Hands wrapper for real-time edge hand landmark tracking.

    Processes raw BGR video frames, mirrors them horizontally for intuitive user
    interaction, draws landmark overlays, and extracts normalized and pixel-space
    coordinates for key kinematic landmarks.

    Attributes:
        static_image_mode (bool): Whether to treat input frames as static images.
        max_num_hands (int): Maximum number of hands to detect and track.
        min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for detection.
        min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for tracking.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ) -> None:
        """
        Initialize the HandTracker with MediaPipe Hands solution.

        Args:
            static_image_mode: If False, optimizes for streaming video.
            max_num_hands: Number of hands to track (default: 1 for unilateral motor assessment).
            min_detection_confidence: Minimum detection confidence threshold (default: 0.7).
            min_tracking_confidence: Minimum tracking confidence threshold (default: 0.7).
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        # Drawing specifications for high-visibility clinical feedback
        self._landmark_spec = self._mp_drawing.DrawingSpec(
            color=(0, 255, 128),  # Vibrant mint green for landmarks
            thickness=2,
            circle_radius=3,
        )
        self._connection_spec = self._mp_drawing.DrawingSpec(
            color=(255, 200, 0),  # Cyan / warm blue for skeleton connections
            thickness=2,
            circle_radius=1,
        )

    def process_frame(
        self, frame: np.ndarray, flip_horizontal: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Process an input BGR frame, extract hand keypoints, and draw annotations.

        Args:
            frame: Raw or pre-mirrored BGR image array from camera or video stream.
            flip_horizontal: If True, applies horizontal mirror flip. Defaults to False
                under standard pipeline design where frame is unconditionally
                mirrored at source ingest to prevent lateral flips on hand transition.

        Returns:
            Tuple of (annotated_frame, landmark_data):
                - annotated_frame (np.ndarray): Frame with drawn landmarks and connections.
                - landmark_data (Optional[Dict[str, Any]]): Dictionary with structured data if
                  a hand is detected, else None. Contains:
                    - 'landmarks_norm': Dict of key landmark indices (0, 4, 8, 12) to (x, y, z).
                    - 'landmarks_pixel': Dict of key landmark indices to (x_px, y_px).
                    - 'handedness': HandednessInfo containing label ("Left"/"Right") & score.
                    - 'timestamp': High-precision monotonic system timestamp.
        """
        if frame is None or frame.size == 0:
            return frame, None

        # Monotonic timestamp for time-synchronized signal processing
        timestamp = time.perf_counter()

        # 1. Flip horizontally only if explicitly requested; otherwise use frame directly.
        # Video frame mirror effect (selfie mode) is applied strictly once unconditionally at capture ingest.
        proc_frame = cv2.flip(frame, 1) if flip_horizontal else frame.copy()
        h, w = proc_frame.shape[:2]

        # 2. Convert BGR to RGB for MediaPipe inference
        rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # 3. Process via MediaPipe Hands
        results = self._hands.process(rgb_frame)

        # 4. Prepare annotated output frame
        annotated_frame = proc_frame.copy()

        # Check if any hand landmarks were detected
        if not results.multi_hand_landmarks:
            return annotated_frame, None

        # Extract primary hand landmarks
        primary_hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks and connections
        self._mp_drawing.draw_landmarks(
            annotated_frame,
            primary_hand_landmarks,
            self._mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=self._landmark_spec,
            connection_drawing_spec=self._connection_spec,
        )

        # Extract Handedness ("Left" / "Right" and confidence score)
        handedness_label = "Unknown"
        handedness_score = 1.0
        if results.multi_handedness and len(results.multi_handedness) > 0:
            classification = results.multi_handedness[0].classification[0]
            handedness_label = classification.label
            handedness_score = float(classification.score)

        handedness_info = HandednessInfo(label=handedness_label, score=handedness_score)

        # Extract normalized coordinates (x, y, z) and pixel coordinates (x_px, y_px)
        landmarks_norm: Dict[Union[int, str], Tuple[float, float, float]] = {}
        landmarks_pixel: Dict[Union[int, str], Tuple[int, int]] = {}

        semantic_names = {
            WRIST_IDX: "wrist",
            THUMB_TIP_IDX: "thumb_tip",
            INDEX_TIP_IDX: "index_tip",
            MIDDLE_MCP_IDX: "middle_mcp",
            MIDDLE_TIP_IDX: "middle_tip",
        }

        for idx in KEY_LANDMARK_INDICES:
            lm = primary_hand_landmarks.landmark[idx]
            norm_coord = (float(lm.x), float(lm.y), float(lm.z))
            px_coord = (int(round(lm.x * w)), int(round(lm.y * h)))

            # Store by integer index
            landmarks_norm[idx] = norm_coord
            landmarks_pixel[idx] = px_coord

            # Also provide semantic name aliases for readable access
            alias = semantic_names[idx]
            landmarks_norm[alias] = norm_coord
            landmarks_pixel[alias] = px_coord

        # Compute scale-normalization anatomical scale factor using strictly 2D Euclidean distance (x, y):
        # Scale Factor = sqrt((x_MCP9 - x_wrist)^2 + (y_MCP9 - y_wrist)^2)
        wrist_2d = np.array(landmarks_norm[WRIST_IDX][:2], dtype=np.float64)
        middle_mcp_2d = np.array(landmarks_norm[MIDDLE_MCP_IDX][:2], dtype=np.float64)
        thumb_2d = np.array(landmarks_norm[THUMB_TIP_IDX][:2], dtype=np.float64)
        index_2d = np.array(landmarks_norm[INDEX_TIP_IDX][:2], dtype=np.float64)

        scale_factor_raw = float(np.linalg.norm(middle_mcp_2d - wrist_2d))
        scale_factor = max(scale_factor_raw, 1e-6)

        # Normalized Aperture = sqrt((x_thumb - x_index)^2 + (y_thumb - y_index)^2) / Scale Factor
        raw_dist_2d = float(np.linalg.norm(thumb_2d - index_2d))
        normalized_aperture = float(raw_dist_2d / scale_factor)

        landmark_data: Dict[str, Any] = {
            "landmarks_norm": landmarks_norm,
            "landmarks_pixel": landmarks_pixel,
            "handedness": handedness_info,
            "timestamp": timestamp,
            "l_ref": scale_factor,
            "scale_factor": scale_factor,
            "raw_distance": raw_dist_2d,
            "normalized_aperture": normalized_aperture,
            "aperture": normalized_aperture,
        }

        return annotated_frame, landmark_data

    @staticmethod
    def create_capture(
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
    ) -> cv2.VideoCapture:
        """
        Initialize OpenCV VideoCapture device with DirectShow (on Windows),
        MJPG FourCC codec for optimal framerate/bandwidth, and explicitly
        enforced 640x480 resolution.

        Args:
            camera_id: Video device index (default: 0).
            width: Desired frame width in pixels (enforced to 640).
            height: Desired frame height in pixels (enforced to 480).

        Returns:
            Configured cv2.VideoCapture object.
        """
        import sys

        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(camera_id)
        else:
            cap = cv2.VideoCapture(camera_id)

        if cap.isOpened():
            # Optimize capture framerate on Windows DirectShow via MJPG FourCC codec
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        return cap

    def close(self) -> None:
        """Release underlying MediaPipe resource handles."""
        if hasattr(self, "_hands") and self._hands is not None:
            self._hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class KinematicBuffer:
    """
    Thread-safe rolling FIFO buffer for kinematic signal processing.

    Stores continuous 3D coordinate trajectories and timestamps. Pops the oldest
    entry when full to maintain a fixed sliding window for spectral decomposition
    (FFT / PSD) and bradykinesia decrement tracking.

    Attributes:
        buffer_size (int): Maximum capacity of the rolling buffer.
    """

    def __init__(self, buffer_size: int = 150) -> None:
        """
        Initialize the KinematicBuffer instance.

        Args:
            buffer_size: Maximum rolling window size in frames (default: 150 frames,
                         corresponding to ~5 seconds at 30 FPS).
        """
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")

        self.buffer_size: int = buffer_size
        self._lock = threading.Lock()
        self._queue: deque = deque(maxlen=self.buffer_size)

    def push(
        self,
        timestamp: float,
        thumb_tip: Union[Tuple[float, ...], np.ndarray],
        index_tip: Union[Tuple[float, ...], np.ndarray],
        wrist: Union[Tuple[float, ...], np.ndarray],
        middle_mcp: Optional[Union[Tuple[float, ...], np.ndarray]] = None,
    ) -> None:
        """
        Store a timestamp and 3D coordinate tuples into the FIFO buffer.

        Automatically pops the oldest record when maximum capacity is reached.

        Args:
            timestamp: Monotonic sample timestamp in seconds.
            thumb_tip: (x, y, z) 3D coordinate of the thumb tip.
            index_tip: (x, y, z) 3D coordinate of the index fingertip.
            wrist: (x, y, z) 3D coordinate of the wrist.
            middle_mcp: Optional (x, y, z) 3D coordinate of the middle finger MCP joint
                        used for scale normalization.
        """
        thumb_tuple = (float(thumb_tip[0]), float(thumb_tip[1]), float(thumb_tip[2]))
        index_tuple = (float(index_tip[0]), float(index_tip[1]), float(index_tip[2]))
        wrist_tuple = (float(wrist[0]), float(wrist[1]), float(wrist[2]))
        mcp_tuple = (
            (float(middle_mcp[0]), float(middle_mcp[1]), float(middle_mcp[2]))
            if middle_mcp is not None
            else None
        )

        entry = (float(timestamp), thumb_tuple, index_tuple, wrist_tuple, mcp_tuple)

        with self._lock:
            self._queue.append(entry)

    def get_time_series(self) -> Dict[str, np.ndarray]:
        """
        Export synchronized time-series arrays from the buffer contents.

        Returns:
            Dictionary containing synchronized NumPy arrays:
                - 'time': 1D array of strictly monotonically increasing relative timestamps
                  in seconds (starting at t = 0.0), shape (N,).
                - 'timestamps': Alias for 'time'.
                - 'thumb_tip': 2D array of (x, y, z) coordinates, shape (N, 3).
                - 'index_tip': 2D array of (x, y, z) coordinates, shape (N, 3).
                - 'wrist': 2D array of (x, y, z) coordinates, shape (N, 3).
                - 'middle_mcp': 2D array of (x, y, z) coordinates, shape (N, 3).
                - 'euclidean_distance': 1D array of 3D Euclidean distances between thumb tip
                  and index tip across all frames, shape (N,).
                - 'raw_distance': Alias for 'euclidean_distance'.
                - 'l_ref': 1D array of scale-normalization reference lengths (Wrist to Middle MCP).
                - 'normalized_aperture': 1D array of scale-invariant finger apertures A(t).
                - 'aperture': Alias for 'normalized_aperture'.
                - 'distance': Scale-invariant normalized aperture for downstream biomarker analysis.
        """
        with self._lock:
            if not self._queue:
                return {
                    "time": np.empty((0,), dtype=np.float64),
                    "timestamps": np.empty((0,), dtype=np.float64),
                    "thumb_tip": np.empty((0, 3), dtype=np.float64),
                    "index_tip": np.empty((0, 3), dtype=np.float64),
                    "wrist": np.empty((0, 3), dtype=np.float64),
                    "middle_mcp": np.empty((0, 3), dtype=np.float64),
                    "euclidean_distance": np.empty((0,), dtype=np.float64),
                    "raw_distance": np.empty((0,), dtype=np.float64),
                    "l_ref": np.empty((0,), dtype=np.float64),
                    "normalized_aperture": np.empty((0,), dtype=np.float64),
                    "aperture": np.empty((0,), dtype=np.float64),
                    "distance": np.empty((0,), dtype=np.float64),
                }

            timestamps = [entry[0] for entry in self._queue]
            thumb = [entry[1] for entry in self._queue]
            index = [entry[2] for entry in self._queue]
            wrist = [entry[3] for entry in self._queue]
            mcp = [entry[4] for entry in self._queue]

        t_arr = np.array(timestamps, dtype=np.float64)
        time_rel = t_arr - t_arr[0]

        # Enforce strictly monotonically increasing relative seconds (t - t0)
        for i in range(1, len(time_rel)):
            if time_rel[i] <= time_rel[i - 1]:
                time_rel[i] = time_rel[i - 1] + 1e-4

        thumb_arr = np.array(thumb, dtype=np.float64)
        index_arr = np.array(index, dtype=np.float64)
        wrist_arr = np.array(wrist, dtype=np.float64)

        # Strictly 2D Euclidean distance between thumb tip and index tip in (x, y):
        # d_2d = sqrt((x_thumb - x_index)^2 + (y_thumb - y_index)^2)
        diff_2d = thumb_arr[:, :2] - index_arr[:, :2]
        euclidean_dist = np.linalg.norm(diff_2d, axis=1)

        # Anatomical Scale Factor = sqrt((x_MCP9 - x_wrist)^2 + (y_MCP9 - y_wrist)^2)
        if any(pt is not None for pt in mcp):
            mcp_cleaned = [pt if pt is not None else wrist[i] for i, pt in enumerate(mcp)]
            mcp_arr = np.array(mcp_cleaned, dtype=np.float64)
            mcp_diff_2d = wrist_arr[:, :2] - mcp_arr[:, :2]
            raw_scale = np.linalg.norm(mcp_diff_2d, axis=1)
            scale_factor = np.maximum(raw_scale, 1e-6)
            normalized_aperture = euclidean_dist / scale_factor
        else:
            mcp_arr = np.empty((0, 3), dtype=np.float64)
            scale_factor = np.ones_like(euclidean_dist)
            normalized_aperture = euclidean_dist.copy()

        return {
            "time": time_rel,
            "timestamps": time_rel,
            "thumb_tip": thumb_arr,
            "index_tip": index_arr,
            "wrist": wrist_arr,
            "middle_mcp": mcp_arr,
            "euclidean_distance": euclidean_dist,
            "raw_distance": euclidean_dist,
            "l_ref": scale_factor,
            "scale_factor": scale_factor,
            "normalized_aperture": normalized_aperture,
            "aperture": normalized_aperture,
            "distance": normalized_aperture,
        }

    def is_ready(self, min_samples: int = 45) -> bool:
        """
        Check if buffer contains sufficient continuous samples for analysis.

        Args:
            min_samples: Minimum number of samples required (default: 45, ~1.5s at 30 FPS).

        Returns:
            True if current sample count is >= min_samples, False otherwise.
        """
        with self._lock:
            return len(self._queue) >= min_samples

    def clear(self) -> None:
        """Reset the internal queues and flush all stored trajectory points."""
        with self._lock:
            self._queue.clear()

    def __len__(self) -> int:
        """Return the current number of frames stored in the buffer."""
        with self._lock:
            return len(self._queue)

    @property
    def current_size(self) -> int:
        """Get the current sample count."""
        return len(self)


class VisionPipeline:
    """
    High-level facade orchestrating camera frame ingestion, landmark tracking,
    and kinematic buffer maintenance for the NeuroMotion system.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        tracking_mode: str = "hands",
        buffer_size: int = 150,
    ) -> None:
        """
        Initialize the VisionPipeline instance.

        Args:
            confidence_threshold: Minimum detection and tracking confidence.
            tracking_mode: Active tracking modality (default: 'hands').
            buffer_size: Capacity of the rolling kinematic buffer.
        """
        self.confidence_threshold = confidence_threshold
        self.tracking_mode = tracking_mode
        self.tracker = HandTracker(
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
        )
        self.buffer = KinematicBuffer(buffer_size=buffer_size)

    def process_frame(
        self, frame: np.ndarray, flip_horizontal: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Process an input video frame, extract hand keypoints, and push to kinematic buffer.

        Args:
            frame: Input image array (BGR).
            flip_horizontal: Whether to apply mirror flip during frame processing.

        Returns:
            Tuple of (annotated_frame, landmark_data).
        """
        annotated_frame, data = self.tracker.process_frame(frame, flip_horizontal=flip_horizontal)
        if data is not None:
            norm = data["landmarks_norm"]
            self.buffer.push(
                timestamp=data["timestamp"],
                thumb_tip=norm[THUMB_TIP_IDX],
                index_tip=norm[INDEX_TIP_IDX],
                wrist=norm[WRIST_IDX],
                middle_mcp=norm[MIDDLE_MCP_IDX],
            )
        return annotated_frame, data

    def get_kinematic_series(self) -> Dict[str, np.ndarray]:
        """Retrieve synchronized kinematic time series from the internal buffer."""
        return self.buffer.get_time_series()

    def close(self) -> None:
        """Clean up resources."""
        self.tracker.close()


def draw_waveform_canvas(
    frame: np.ndarray,
    distance_series: np.ndarray,
    peak_indices: np.ndarray,
    origin: Tuple[int, int],
    size: Tuple[int, int],
) -> None:
    """
    Render a clean real-time mini waveform monitor onto an OpenCV frame.

    Args:
        frame: Destination BGR image array.
        distance_series: 1D array of rolling finger aperture values.
        peak_indices: Frame indices of detected tap peaks.
        origin: (x, y) top-left coordinates of the monitor widget.
        size: (width, height) of the monitor widget.
    """
    x, y = origin
    w, h = size
    h_frame, w_frame = frame.shape[:2]

    if x + w > w_frame or y + h > h_frame:
        return

    # Semi-transparent background card
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (14, 18, 26), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Cyan/slate border and grid line
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 220), 1, cv2.LINE_AA)
    mid_y = y + h // 2
    cv2.line(frame, (x, mid_y), (x + w, mid_y), (35, 45, 60), 1, cv2.LINE_AA)

    # Title label
    cv2.putText(
        frame,
        "Norm Aperture A(t)",
        (x + 8, y + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (180, 220, 240),
        1,
        cv2.LINE_AA,
    )

    n_pts = len(distance_series)
    if n_pts < 2:
        return

    min_val = 0.0
    max_val = max(0.60, float(np.max(distance_series)))
    val_range = max_val - min_val

    margin_x = 8
    margin_y = 20
    usable_w = w - 2 * margin_x
    usable_h = h - margin_y - 8

    pts = []
    for i, val in enumerate(distance_series):
        pt_x = int(x + margin_x + (i / (n_pts - 1)) * usable_w)
        norm_y = (val - min_val) / val_range
        norm_y = max(0.0, min(1.0, norm_y))
        pt_y = int(y + h - 6 - norm_y * usable_h)
        pts.append((pt_x, pt_y))

    pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        frame, [pts_arr], isClosed=False, color=(0, 255, 180), thickness=2, lineType=cv2.LINE_AA
    )

    # Highlight detected peaks with bright emerald markers
    peak_set = set(peak_indices.tolist()) if len(peak_indices) > 0 else set()
    for idx in peak_set:
        if 0 <= idx < len(pts):
            cv2.circle(frame, pts[idx], 4, (0, 220, 255), -1, cv2.LINE_AA)


def draw_clinical_hud(
    frame: np.ndarray,
    fps: float,
    buffer_len: int,
    buffer_capacity: int,
    handedness_str: str,
    current_aperture: float,
    current_state: str,
    session_taps: int,
    instantaneous_cadence: float,
    trial_status: Dict[str, Any],
    tap_metrics: Optional[Dict[str, Any]] = None,
    updrs_tapping: Optional[Dict[str, Any]] = None,
    tremor_metrics: Optional[Dict[str, Any]] = None,
    updrs_tremor: Optional[Dict[str, Any]] = None,
    distance_series: Optional[np.ndarray] = None,
    scale_factor: float = 0.0,
) -> np.ndarray:
    """
    Render a polished medical Heads-Up Display (HUD) overlay onto the OpenCV frame.

    Guarantees non-overlapping, line-spaced semi-transparent diagnostic cards:
      - Top Left Card: Live FPS, Hand Tracking Status, Current Scale Ratio.
      - Trial Overlay Card: Active Trial Status (Timer countdown, Tap Count, Real-time Cadence).
      - Live Telemetry Card: Session Tap Count, Instantaneous Cadence, Tap State & Aperture Bar.
      - Mini Waveform Monitor: Bottom-right corner without visual occlusion.

    Args:
        frame: Annotated BGR frame (e.g. 640x480).
        fps: Measured frame acquisition rate.
        buffer_len: Current sample count in rolling buffer.
        buffer_capacity: Maximum capacity of rolling buffer.
        handedness_str: String representation of handedness ("Right (0.98)").
        current_aperture: Instantaneous scale-normalized aperture A(t).
        current_state: Active tap state machine state (TOUCH, OPENING, PEAK, CLOSING).
        session_taps: Strictly non-decreasing session tap counter.
        instantaneous_cadence: Live cadence (Hz) from the last 5 confirmed taps.
        trial_status: Dictionary from TappingTrialManager (state, remaining_time, metrics, score).
        tap_metrics: Optional rolling tap metrics.
        updrs_tapping: Optional live MDS-UPDRS Task 3.4 score dictionary.
        tremor_metrics: Optional spectral tremor decomposition metrics.
        updrs_tremor: Optional clinical tremor rating.
        distance_series: Optional rolling time-series of finger aperture.
        scale_factor: Anatomical scale factor L_ref (Wrist to Middle MCP).

    Returns:
        Frame with polished clinical telemetry HUD.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    tap_metrics = tap_metrics or {}
    updrs_tapping = updrs_tapping or {}
    tremor_metrics = tremor_metrics or {}
    updrs_tremor = updrs_tremor or {}

    # =========================================================================
    # CARD 1: TOP-LEFT DIAGNOSTIC CARD (Non-overlapping: x=12..232, y=10..86)
    # =========================================================================
    c1_x, c1_y = 12, 10
    c1_w, c1_h = 220, 76
    c1_overlay = frame.copy()
    cv2.rectangle(c1_overlay, (c1_x, c1_y), (c1_x + c1_w, c1_y + c1_h), (12, 16, 24), -1)
    cv2.addWeighted(c1_overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (c1_x, c1_y), (c1_x + c1_w, c1_y + c1_h), (40, 55, 75), 1, cv2.LINE_AA)

    # Line 1: Live FPS & Brand
    fps_color = (0, 255, 128) if fps >= 25.0 else ((0, 200, 255) if fps >= 18.0 else (0, 80, 255))
    cv2.putText(frame, f"FPS: {fps:4.1f}", (c1_x + 10, c1_y + 20), font, 0.44, fps_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "| NeuroMotion", (c1_x + 95, c1_y + 20), font, 0.40, (160, 210, 230), 1, cv2.LINE_AA)

    # Line 2: Hand Tracking Status
    is_tracked = bool(handedness_str and handedness_str != "None" and "Unknown" not in handedness_str)
    hand_color = (0, 255, 200) if is_tracked else (0, 165, 255)
    hand_status_txt = f"Hand: {handedness_str}" if is_tracked else "Hand: Searching..."
    cv2.putText(frame, hand_status_txt, (c1_x + 10, c1_y + 43), font, 0.40, hand_color, 1, cv2.LINE_AA)

    # Line 3: Current Scale Ratio
    scale_val = scale_factor if scale_factor > 0 else 0.0
    cv2.putText(frame, f"Scale: {scale_val:.3f} L_ref (2D)", (c1_x + 10, c1_y + 65), font, 0.38, (240, 240, 240), 1, cv2.LINE_AA)

    # =========================================================================
    # CARD 2: TRIAL OVERLAY CARD (Top-Right / Middle: x=242..628, y=10..86)
    # =========================================================================
    trial_state = trial_status.get("state", "IDLE")
    rem_time = trial_status.get("remaining_time", 10.0)
    trial_taps = trial_status.get("trial_taps_count", 0)

    c2_x, c2_y = 242, 10
    c2_w = w - c2_x - 12
    c2_h = 76

    c2_overlay = frame.copy()
    cv2.rectangle(c2_overlay, (c2_x, c2_y), (c2_x + c2_w, c2_y + c2_h), (14, 18, 28), -1)
    cv2.addWeighted(c2_overlay, 0.85, frame, 0.15, 0, frame)

    if trial_state == "IDLE":
        cv2.rectangle(frame, (c2_x, c2_y), (c2_x + c2_w, c2_y + c2_h), (0, 180, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "MDS-UPDRS 3.4 TRIAL: IDLE", (c2_x + 10, c2_y + 22), font, 0.46, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press SPACE or 'R' to Start 10s Formal Trial", (c2_x + 10, c2_y + 44), font, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, "Tap thumb & index as fast & wide as possible", (c2_x + 10, c2_y + 64), font, 0.33, (150, 175, 195), 1, cv2.LINE_AA)

    elif trial_state == "ACTIVE":
        cv2.rectangle(frame, (c2_x, c2_y), (c2_x + c2_w, c2_y + c2_h), (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"TRIAL ACTIVE: {rem_time:4.1f}s REMAINING", (c2_x + 10, c2_y + 22), font, 0.46, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Taps: {trial_taps}  |  Cadence: {instantaneous_cadence:4.2f} Hz", (c2_x + 10, c2_y + 42), font, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # 10s Progress Bar
        bar_x = c2_x + 10
        bar_y = c2_y + 52
        bar_w = c2_w - 20
        bar_h = 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 42, 55), -1)
        fill_w = int(round((10.0 - rem_time) / 10.0 * bar_w))
        fill_w = max(0, min(bar_w, fill_w))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 220, 255), -1)

    elif trial_state == "COMPLETED":
        trial_score_dict = trial_status.get("score") or updrs_tapping
        trial_metrics_dict = trial_status.get("metrics") or tap_metrics
        t_score = trial_score_dict.get("score", 0)
        t_label = trial_score_dict.get("label", "Unknown")

        score_colors = {
            0: (0, 255, 128),  # Normal (Green)
            1: (0, 220, 255),  # Slight (Cyan)
            2: (0, 165, 255),  # Mild (Orange)
            3: (50, 80, 255),  # Moderate (Red-Orange)
            4: (0, 0, 255),    # Severe (Red)
        }
        b_color = score_colors.get(t_score, (0, 255, 128))
        cv2.rectangle(frame, (c2_x, c2_y), (c2_x + c2_w, c2_y + c2_h), b_color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"TRIAL RESULT -> SCORE {t_score} ({t_label})", (c2_x + 10, c2_y + 22), font, 0.44, b_color, 2, cv2.LINE_AA)
        rate_val = trial_metrics_dict.get("tap_rate_hz", 0.0)
        slope_val = trial_metrics_dict.get("amplitude_decay_slope", 0.0)
        cv_val = trial_metrics_dict.get("rhythm_cv", 0.0)
        cv2.putText(frame, f"Taps: {trial_taps} in 10s | Rate: {rate_val:.2f} Hz | Slope: {slope_val:+.3f}", (c2_x + 10, c2_y + 44), font, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Rhythm CV: {cv_val:.2f}  (Press SPACE to re-test)", (c2_x + 10, c2_y + 64), font, 0.34, (180, 210, 235), 1, cv2.LINE_AA)

    # =========================================================================
    # CARD 3: LIVE PERSISTENT TELEMETRY (Non-overlapping: x=12..232, y=94..246)
    # =========================================================================
    c3_x, c3_y = 12, 94
    c3_w, c3_h = 220, 152
    c3_overlay = frame.copy()
    cv2.rectangle(c3_overlay, (c3_x, c3_y), (c3_x + c3_w, c3_y + c3_h), (12, 16, 24), -1)
    cv2.addWeighted(c3_overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (c3_x, c3_y), (c3_x + c3_w, c3_y + c3_h), (40, 55, 75), 1, cv2.LINE_AA)

    # Session Tap Count
    cv2.putText(frame, "Session Taps:", (c3_x + 10, c3_y + 22), font, 0.40, (190, 205, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{session_taps} taps", (c3_x + 120, c3_y + 22), font, 0.46, (0, 255, 200), 2, cv2.LINE_AA)

    # Instantaneous Cadence
    cv2.putText(frame, "Cadence:", (c3_x + 10, c3_y + 46), font, 0.40, (190, 205, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{instantaneous_cadence:4.2f} Hz", (c3_x + 120, c3_y + 46), font, 0.46, (255, 255, 255), 2, cv2.LINE_AA)

    # Tap State
    state_colors = {
        "TOUCH": (0, 255, 128),
        "OPENING": (255, 220, 0),
        "PEAK": (0, 255, 255),
        "CLOSING": (0, 165, 255),
    }
    st_color = state_colors.get(current_state, (200, 200, 200))
    cv2.putText(frame, "Tap State:", (c3_x + 10, c3_y + 70), font, 0.40, (190, 205, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"[{current_state}]", (c3_x + 120, c3_y + 70), font, 0.44, st_color, 2, cv2.LINE_AA)

    # Scale-Normalized Aperture
    cv2.putText(frame, "Aperture A(t):", (c3_x + 10, c3_y + 96), font, 0.40, (190, 205, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{current_aperture:4.2f} L_ref", (c3_x + 120, c3_y + 96), font, 0.44, (255, 220, 100), 2, cv2.LINE_AA)

    # Aperture Mini-Bar Gauge
    g_x = c3_x + 10
    g_y = c3_y + 112
    g_w = c3_w - 20
    g_h = 12
    cv2.rectangle(frame, (g_x, g_y), (g_x + g_w, g_y + g_h), (28, 34, 46), -1)
    fill_len = int(round(min(1.0, current_aperture / 1.0) * g_w))
    fill_len = max(0, min(g_w, fill_len))
    cv2.rectangle(frame, (g_x, g_y), (g_x + fill_len, g_y + g_h), st_color, -1)
    cv2.rectangle(frame, (g_x, g_y), (g_x + g_w, g_y + g_h), (55, 68, 85), 1, cv2.LINE_AA)

    # Touch threshold tick mark (0.22)
    tick_x = g_x + int(0.22 * g_w)
    cv2.line(frame, (tick_x, g_y - 2), (tick_x, g_y + g_h + 2), (0, 255, 128), 2, cv2.LINE_AA)
    cv2.putText(frame, "Touch Thresh: 0.22", (c3_x + 10, c3_y + 140), font, 0.32, (150, 175, 195), 1, cv2.LINE_AA)

    # =========================================================================
    # MINI WAVEFORM MONITOR (Bottom-Right: x=400..628, y=360..456)
    # =========================================================================
    if distance_series is not None and len(distance_series) > 1:
        wave_w = 228
        wave_h = 96
        wave_x = w - wave_w - 12
        wave_y = h - wave_h - 24
        peak_indices = tap_metrics.get("peak_indices", np.empty((0,), dtype=int))
        draw_waveform_canvas(frame, distance_series, peak_indices, (wave_x, wave_y), (wave_w, wave_h))

    # =========================================================================
    # FOOTER INSTRUCTION BAR (y = h - 8)
    # =========================================================================
    footer_text = "SPACE/R: 10s Trial | 'q'/ESC: Exit | 2D Scale-Normalized"
    cv2.putText(frame, footer_text, (12, h - 8), font, 0.38, (180, 200, 220), 1, cv2.LINE_AA)

    return frame


def draw_trial_complete_overlay(
    frame: np.ndarray,
    trial_status: Optional[Dict[str, Any]] = None,
    score_data: Optional[Dict[str, Any]] = None,
    metric_data: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Render a clean, semi-transparent in-frame completion banner onto the final paused OpenCV frame.

    Guarantees that participants in Streamlit fullscreen mode clearly see that
    the 10-second assessment has concluded successfully, preventing confusion.

    Args:
        frame: Destination BGR image array (e.g. 640x480).
        trial_status: Optional trial state information dictionary.
        score_data: Optional MDS-UPDRS scoring dictionary (score, label, clinical_rationale).
        metric_data: Optional kinematic tap metrics dictionary (total_taps, tap_rate_hz).

    Returns:
        Frame with clinical 'TRIAL EVALUATION COMPLETE' overlay.
    """
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    score_data = score_data or {}
    metric_data = metric_data or {}

    # Card dimensions - centered and responsive to frame size
    card_w = min(560, max(300, w - 40))
    card_h = 162
    card_x = max(10, (w - card_w) // 2)
    card_y = max(10, (h - card_h) // 2 + 10)

    # 1. Semi-transparent dark background card (deep medical slate navy)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (card_x, card_y),
        (card_x + card_w, card_y + card_h),
        (12, 16, 26),
        -1,
    )
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    # 2. Sleek Emerald Green accent border & subtle inner border
    cv2.rectangle(
        frame,
        (card_x, card_y),
        (card_x + card_w, card_y + card_h),
        (0, 210, 106),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        frame,
        (card_x + 3, card_y + 3),
        (card_x + card_w - 3, card_y + card_h - 3),
        (35, 50, 70),
        1,
        cv2.LINE_AA,
    )

    # 3. Top status pill badge: 'STATUS: PROCESSED'
    badge_x = card_x + 18
    badge_y = card_y + 14
    badge_w = 175
    badge_h = 24
    cv2.rectangle(
        frame,
        (badge_x, badge_y),
        (badge_x + badge_w, badge_y + badge_h),
        (16, 44, 32),
        -1,
    )
    cv2.rectangle(
        frame,
        (badge_x, badge_y),
        (badge_x + badge_w, badge_y + badge_h),
        (0, 210, 106),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(frame, (badge_x + 12, badge_y + 12), 4, (0, 255, 128), -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "STATUS: PROCESSED",
        (badge_x + 22, badge_y + 16),
        font,
        0.36,
        (0, 255, 150),
        1,
        cv2.LINE_AA,
    )

    # 4. Prominent Header Text (Bright White / Emerald Green)
    cv2.putText(
        frame,
        "TRIAL EVALUATION COMPLETE",
        (card_x + 18, card_y + 68),
        font,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 5. Clear Subtitle (Fullscreen user guidance)
    cv2.putText(
        frame,
        "MDS-UPDRS metrics ready. Minimize / scroll down to inspect report.",
        (card_x + 18, card_y + 96),
        font,
        0.38,
        (210, 230, 248),
        1,
        cv2.LINE_AA,
    )

    # 6. Dividing separator
    cv2.line(
        frame,
        (card_x + 18, card_y + 112),
        (card_x + card_w - 18, card_y + 112),
        (38, 52, 72),
        1,
        cv2.LINE_AA,
    )

    # 7. Assessment Summary Metrics
    t_score = score_data.get("score")
    t_label = score_data.get("label", "Evaluation Ready")
    total_taps = metric_data.get("total_taps", 0)
    rate_hz = metric_data.get("tap_rate_hz", 0.0)

    if t_score is not None:
        score_colors = {
            0: (0, 255, 128),  # Normal
            1: (0, 220, 255),  # Slight
            2: (0, 165, 255),  # Mild
            3: (50, 80, 255),  # Moderate
            4: (0, 0, 255),    # Severe
        }
        res_color = score_colors.get(t_score, (0, 255, 128))
        summary_txt = f"Part III Task 3.4: Score {t_score} ({t_label})  |  {total_taps} Taps ({rate_hz:.2f} Hz)"
    else:
        res_color = (0, 220, 255)
        summary_txt = f"Formal 10s Motor Assessment: {total_taps} Taps ({rate_hz:.2f} Hz) Recorded"

    cv2.putText(
        frame,
        summary_txt,
        (card_x + 18, card_y + 138),
        font,
        0.42,
        res_color,
        1,
        cv2.LINE_AA,
    )

    # Press to restart hint on right side of bottom line if space permits
    if card_w >= 500:
        cv2.putText(
            frame,
            "Press 'Start Trial' to re-test",
            (card_x + card_w - 195, card_y + 138),
            font,
            0.32,
            (148, 163, 184),
            1,
            cv2.LINE_AA,
        )

    return frame

