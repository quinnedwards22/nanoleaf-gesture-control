# gesture.py
import cv2
import mediapipe as mp
import math
import time

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(min_detection_confidence=0.7,
                         min_tracking_confidence=0.7)
mp_draw  = mp.solutions.drawing_utils

# tweak these as you like
MAX_PINCH_DIST   = 200    # px → maps to 100% brightness
UPDATE_INTERVAL  = 0.2    # seconds between API calls

_last_time   = time.time()
_current_bri = 50

def process_frame(frame):
    """
    Input: BGR frame from OpenCV
    Returns: (brightness [0–100], output_frame with overlays)
    """
    global _last_time, _current_bri

    img = cv2.flip(frame, 1)
    h, w, _ = img.shape
    rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res     = hands.process(rgb)

    if res.multi_hand_landmarks:
        lm     = res.multi_hand_landmarks[0].landmark
        thumb  = (int(lm[4].x * w), int(lm[4].y * h))
        index  = (int(lm[8].x * w), int(lm[8].y * h))

        cv2.circle(img, thumb, 8, (0,255,0), cv2.FILLED)
        cv2.circle(img, index, 8, (0,255,0), cv2.FILLED)
        cv2.line(img, thumb, index, (0,255,0), 2)

        dist = math.hypot(index[0]-thumb[0], index[1]-thumb[1])
        bri  = int((dist / MAX_PINCH_DIST) * 100)
        bri  = max(0, min(100, bri))

        now = time.time()
        if now - _last_time > UPDATE_INTERVAL:
            _last_time   = now
            _current_bri = bri

        cv2.putText(img, f"{_current_bri}%", (thumb[0], thumb[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return _current_bri, img
