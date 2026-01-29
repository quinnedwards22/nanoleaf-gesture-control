# gesture.py (MediaPipe Tasks version)
import cv2
import mediapipe as mp
import math
import time
import os

# --- Tuning ---
MAX_PINCH_DIST  = 200   # px -> maps to 100% brightness
UPDATE_INTERVAL = 0.2   # seconds between updates

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


HERE = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(HERE, "hand_landmarker.task")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)
detector = HandLandmarker.create_from_options(options)


_last_time = time.time()
_current_bri = 50

def process_frame(frame):
    """
    Input: BGR frame from OpenCV
    Returns: (brightness [0–100], output_frame with overlays)
    """
    global _last_time, _current_bri

    img = cv2.flip(frame, 1)
    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # MediaPipe expects a monotonically increasing timestamp in ms for VIDEO mode
    ts_ms = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, ts_ms)

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]  # first detected hand

        # thumb tip (4), index tip (8)
        thumb = (int(lm[4].x * w), int(lm[4].y * h))
        index = (int(lm[8].x * w), int(lm[8].y * h))
        thumb_base = (int(lm[2].x * w), int(lm[2].y * h))

        thumb_depth = lm[4].z
        index_depth = lm[8].z

        scaling_factor = math.fabs((1 / index_depth) * 0.01) if index_depth != 0 else 1



        cv2.circle(img, thumb, 8, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, index, 8, (0, 255, 0), cv2.FILLED)
        cv2.line(img, thumb, index, (0, 255, 0), 2)

        
        dist1 = math.hypot(thumb[0] - thumb_base[0], thumb[1] - thumb_base[1])
        dist2 = math.hypot(index[0] - thumb_base[0], index[1] - thumb_base[1])

        new_distance = math.hypot(dist1, dist2)

        dist = math.hypot(index[0] - thumb[0], index[1] - thumb[1]) / scaling_factor
        bri = int((new_distance / MAX_PINCH_DIST) * 100)
        bri = max(0, min(100, bri))


        cv2.putText(
            img,
            f"{dist:.2f} {new_distance:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            1,
        )

        now = time.time()
        if now - _last_time > UPDATE_INTERVAL:
            _last_time = now
            _current_bri = bri

        cv2.putText(
            img,
            f"{_current_bri}%",
            (thumb[0], max(30, thumb[1] - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    return _current_bri, img
