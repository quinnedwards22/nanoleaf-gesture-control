# gesture.py (MediaPipe Tasks version)
import cv2
import mediapipe as mp
import math
import time
import os
import enum

class Hand(enum.Enum):
    LEFT = 1
    RIGHT = 2

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
        lm3d = result.hand_world_landmarks[0]   # Landmark (meters, for math)
        lm2d = result.hand_landmarks[0]         # NormalizedLandmark (for drawing)

        thumb_w = lm3d[4] # tip of thumb
        index_w = lm3d[8] # tip of index finger

        pinky_w = lm3d[20] # tip of pinky finger


        # distance between thumb + index in 3D space (meters)
        pinch_m = math.sqrt(
            (thumb_w.x - index_w.x)**2 +
            (thumb_w.y - index_w.y)**2 +
            (thumb_w.z - index_w.z)**2
        )

        index_mcp = lm3d[5] # base of index finger
        pinky_mcp = lm3d[17] # base of pinky finger

        # distance between index + pinky in 3D space (meters)
        palm_width_m = math.sqrt(
            (index_mcp.x - pinky_mcp.x)**2 +
            (index_mcp.y - pinky_mcp.y)**2 +
            (index_mcp.z - pinky_mcp.z)**2
        )

        pinch_norm = pinch_m / palm_width_m - 0.3 if palm_width_m > 1e-6 else 0 # normalize by palm size

        thumb_px = (int(lm2d[4].x * w), int(lm2d[4].y * h)) # tip of thumb 2d
        index_px = (int(lm2d[8].x * w), int(lm2d[8].y * h)) # tip of index finger 2d

        isPinkyUp = lm2d[20].y < lm2d[18].y  # is pinky finger up?

        # draw
        cv2.circle(img, thumb_px, 8, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, index_px, 8, (0, 255, 0), cv2.FILLED)
        cv2.line(img, thumb_px, index_px, (0, 255, 0), 2)

    

        PINCH_MIN = 0.15  # “fully pinched”
        PINCH_MAX = 1  # “fully open”

        pinch2d = math.sqrt((index_px[0] - thumb_px[0])**2 + (index_px[1] - thumb_px[1])**2) / 100
       
        if pinch2d > PINCH_MIN:  # avoid noise when very close
            ti = (pinch_norm - PINCH_MIN) / (PINCH_MAX - PINCH_MIN)
        else:
            ti = 0.0
        
        if isPinkyUp:
            t = max(0.0, min(1.0, ti))
            bri = int(t * 100)
        else:
            bri = -1  # do not update brightness if pinky is down

        cv2.putText(
            img,
            f"{pinch_m:.2f} {palm_width_m:.2f} {pinch2d:.2f} {bri}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            1,
        )
        cv2.putText(
            img,
            f"{isPinkyUp}%",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            1,
        )

        now = time.time()
        if now - _last_time > UPDATE_INTERVAL:
            _last_time = now
            _current_bri = bri

    return _current_bri, img
