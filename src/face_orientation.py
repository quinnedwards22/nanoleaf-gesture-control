# face_orientation.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_PATH = (HERE / "models" / "face_landmarker.task").resolve()

# ==========
# Config
# ==========

# Orientation rule example:
# "happening" when yaw is to the RIGHT beyond this threshold (degrees)
YAW_MIN = -2
YAW_MAX = 7

# Optional: also require roll within a range (to avoid tilted head false positives)
MAX_ABS_ROLL_DEG = 20.0

# If face presence score is below this, treat as "no reliable face"
MIN_FACE_PRESENCE = 0.5

# Landmark indices (MediaPipe Face Mesh index space)
IDX_NOSE_TIP = 1
IDX_CHIN = 152
IDX_LEFT_EYE_OUTER = 33
IDX_RIGHT_EYE_OUTER = 263

# ==========
# MediaPipe Task Types
# ==========
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_landmarker: Optional[FaceLandmarker] = None


@dataclass
class HeadPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


def _ensure_landmarker() -> FaceLandmarker:
    global _landmarker
    if _landmarker is not None:
        return _landmarker

    print("CWD:", Path.cwd())
    print("MODEL_PATH:", MODEL_PATH)
    print("MODEL_EXISTS:", MODEL_PATH.exists())

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Unable to find face model at: {MODEL_PATH}\n"
            f"Make sure 'models/face_landmarker.task' exists next to face_orientation.py"
        )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _landmarker = FaceLandmarker.create_from_options(options)
    return _landmarker



def _to_mp_image(bgr_frame: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _get_landmark_px(lms, idx: int, w: int, h: int) -> np.ndarray:
    """Convert normalized landmark to pixel coords (x,y)."""
    lm = lms[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _compute_head_pose_2d(
    face_landmarks,
    w: int,
    h: int,
) -> HeadPose:
    """
    Lightweight pose estimate from a few landmarks.
    Not a full PnP solve—good enough for thresholds/detection.

    - roll: angle of eye line
    - yaw: nose horizontal displacement relative to eye midpoint
    - pitch: nose vertical displacement relative to eye midpoint vs chin
    """
    left_eye = _get_landmark_px(face_landmarks, IDX_LEFT_EYE_OUTER, w, h)
    right_eye = _get_landmark_px(face_landmarks, IDX_RIGHT_EYE_OUTER, w, h)
    nose = _get_landmark_px(face_landmarks, IDX_NOSE_TIP, w, h)
    chin = _get_landmark_px(face_landmarks, IDX_CHIN, w, h)

    eye_mid = (left_eye + right_eye) * 0.5
    eye_vec = right_eye - left_eye

    # Roll: tilt angle of eye line
    roll_rad = math.atan2(eye_vec[1], eye_vec[0])
    roll_deg = math.degrees(roll_rad)

    # Normalize by inter-eye distance to be distance-invariant
    eye_dist = float(np.linalg.norm(eye_vec)) + 1e-6

    # Yaw: nose offset left/right from eye midpoint
    yaw_norm = (nose[0] - eye_mid[0]) / eye_dist
    # Empirical scale to degrees-ish (tweak for your camera)
    yaw_deg = float(yaw_norm * 35.0)

    # Pitch: compare vertical nose position relative to eyes and chin
    chin_dist = float(np.linalg.norm(chin - eye_mid)) + 1e-6
    pitch_norm = (nose[1] - eye_mid[1]) / chin_dist
    pitch_deg = float(pitch_norm * 40.0)

    return HeadPose(yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg)


def _rule_is_happening(pose: HeadPose) -> Tuple[bool, float]:
    """
    Define your orientation condition here.
    Returns: (is_happening, confidence 0..1)
    """
    # Example: "happening" when looking right enough and not too tilted
    yaw_ok = YAW_MIN <= pose.yaw_deg <= YAW_MAX
    roll_ok = abs(pose.roll_deg) <= MAX_ABS_ROLL_DEG

    is_happening = bool(yaw_ok and roll_ok)

    # Confidence: distance past threshold, clipped 0..1
    # If not happening, confidence still indicates closeness.
    yaw_conf = (pose.yaw_deg - YAW_MIN) / (YAW_MAX - YAW_MIN)
    yaw_conf = float(np.clip(yaw_conf, 0.0, 1.0))

    roll_conf = 1.0 - float(np.clip(abs(pose.roll_deg) / MAX_ABS_ROLL_DEG, 0.0, 1.0))

    confidence = float(np.clip(0.7 * yaw_conf + 0.3 * roll_conf, 0.0, 1.0))
    return is_happening, confidence


def process_frame(frame_bgr: np.ndarray) -> Tuple[bool, float, Dict[str, float], np.ndarray]:
    """
    Main entry:
      returns (is_happening, confidence, pose_dict, vis_frame)

    - is_happening: bool
    - confidence: float 0..1
    - pose_dict: {"yaw_deg":..., "pitch_deg":..., "roll_deg":...}
    - vis_frame: annotated BGR frame
    """
    landmarker = _ensure_landmarker()
    h, w = frame_bgr.shape[:2]
    vis = frame_bgr.copy()

    mp_img = _to_mp_image(frame_bgr)
    ts_ms = int(time.time() * 1000)

    result = landmarker.detect_for_video(mp_img, ts_ms)

    if not result.face_landmarks or len(result.face_landmarks) == 0:
        _draw_status(vis, "No face", (0, 0, 255))
        return False, 0.0, {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}, vis

    face_landmarks = result.face_landmarks[0]

    # Face presence score can be available depending on model/config;
    # if absent, we assume ok.
    presence_ok = True
    presence_score = None
    if hasattr(result, "face_presence") and result.face_presence:
        presence_score = float(result.face_presence[0])
        presence_ok = presence_score >= MIN_FACE_PRESENCE

    if not presence_ok:
        _draw_status(vis, f"Low presence ({presence_score:.2f})", (0, 0, 255))
        return False, 0.0, {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}, vis

    pose = _compute_head_pose_2d(face_landmarks, w, h)
    happening, conf = _rule_is_happening(pose)

    # Draw a few landmarks + text
    _draw_debug(vis, face_landmarks, w, h, pose, happening, conf)

    return happening, conf, {"yaw_deg": pose.yaw_deg, "pitch_deg": pose.pitch_deg, "roll_deg": pose.roll_deg}, vis


def _draw_status(img: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def _draw_debug(
    img: np.ndarray,
    face_landmarks,
    w: int,
    h: int,
    pose: HeadPose,
    happening: bool,
    conf: float,
) -> None:
    # Draw key points
    for idx in (IDX_LEFT_EYE_OUTER, IDX_RIGHT_EYE_OUTER, IDX_NOSE_TIP, IDX_CHIN):
        p = _get_landmark_px(face_landmarks, idx, w, h).astype(int)
        cv2.circle(img, tuple(p), 4, (0, 255, 0), -1)

    status = "HAPPENING" if happening else "not happening"
    color = (0, 255, 0) if happening else (0, 0, 255)

    _draw_status(img, f"{status}  conf={conf:.2f}", color)
    cv2.putText(
        img,
        f"yaw={pose.yaw_deg:.1f}  pitch={pose.pitch_deg:.1f}  roll={pose.roll_deg:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
