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
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Process a raw BGR frame from OpenCV, perform mirror flip, and extract keypoints.

        Args:
            frame: Raw BGR image array from camera or video stream.

        Returns:
            Tuple of (annotated_frame, landmark_data):
                - annotated_frame (np.ndarray): Horizontally flipped frame with drawn landmarks.
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

        # 1. Flip horizontally for natural mirror interaction
        flipped_frame = cv2.flip(frame, 1)
        h, w = flipped_frame.shape[:2]

        # 2. Convert BGR to RGB for MediaPipe inference
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # 3. Process via MediaPipe Hands
        results = self._hands.process(rgb_frame)

        # 4. Prepare annotated output frame
        annotated_frame = flipped_frame.copy()

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

        # Compute scale-normalization reference length L_ref:
        # Euclidean distance between Landmark 0 (Wrist) and Landmark 9 (Middle Finger MCP)
        wrist_norm = np.array(landmarks_norm[WRIST_IDX], dtype=np.float64)
        middle_mcp_norm = np.array(landmarks_norm[MIDDLE_MCP_IDX], dtype=np.float64)
        thumb_norm = np.array(landmarks_norm[THUMB_TIP_IDX], dtype=np.float64)
        index_norm = np.array(landmarks_norm[INDEX_TIP_IDX], dtype=np.float64)

        l_ref = float(np.linalg.norm(wrist_norm - middle_mcp_norm))
        if l_ref < 1e-4:
            l_ref = 1.0

        # Compute raw Euclidean distance & scale-invariant normalized aperture A(t) = ||thumb - index|| / L_ref
        raw_dist = float(np.linalg.norm(thumb_norm - index_norm))
        normalized_aperture = float(raw_dist / l_ref)

        landmark_data: Dict[str, Any] = {
            "landmarks_norm": landmarks_norm,
            "landmarks_pixel": landmarks_pixel,
            "handedness": handedness_info,
            "timestamp": timestamp,
            "l_ref": l_ref,
            "raw_distance": raw_dist,
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

        # Compute true 3D Euclidean distance between thumb tip and index tip:
        # d = np.linalg.norm(thumb_array - index_array, axis=1) using (x, y, z)
        diff = thumb_arr - index_arr
        euclidean_dist = np.linalg.norm(diff, axis=1)

        # Scale-normalization reference length L_ref = ||p_wrist - p_middle_mcp||_2
        if any(pt is not None for pt in mcp):
            mcp_cleaned = [pt if pt is not None else wrist[i] for i, pt in enumerate(mcp)]
            mcp_arr = np.array(mcp_cleaned, dtype=np.float64)
            l_ref = np.linalg.norm(wrist_arr - mcp_arr, axis=1)
            l_ref = np.where(l_ref > 1e-4, l_ref, 1.0)
            normalized_aperture = euclidean_dist / l_ref
        else:
            mcp_arr = np.empty((0, 3), dtype=np.float64)
            l_ref = np.ones_like(euclidean_dist)
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
            "l_ref": l_ref,
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
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Process an input video frame, extract hand keypoints, and push to kinematic buffer.

        Args:
            frame: Input image array (BGR).

        Returns:
            Tuple of (annotated_frame, landmark_data).
        """
        annotated_frame, data = self.tracker.process_frame(frame)
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
