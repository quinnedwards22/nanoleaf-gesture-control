import cv2
import mediapipe as mp
import requests
import time
import math

# 1. Nanoleaf setup (updated token)
NANO_IP    = "192.168.2.34"
AUTH_TOKEN = "9N4HGFpkLsa1jkBfbWxTdUk5FAzETGyP"
API_BASE   = f"http://{NANO_IP}:16021/api/v1/{AUTH_TOKEN}/state"

# 2. Helper
def set_brightness(bri: int):
    bri = max(0, min(100, bri))
    payload = {"brightness": {"value": bri}}
    resp = requests.put(API_BASE, json=payload)
    resp.raise_for_status()

# 3. Hand tracking init
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(min_detection_confidence=0.7,
                         min_tracking_confidence=0.7)
mp_draw  = mp.solutions.drawing_utils
cap      = cv2.VideoCapture(0)

# 4. Brightness state
last_update = time.time()
brightness  = 50
set_brightness(brightness)

# 5. Mapping parameters
MAX_PINCH_DIST = 200  # pixels → adjust after testing

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res     = hands.process(img_rgb)

    thumb = index = None

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark

        # get pixel coords for tip of thumb (4) and index (8)
        thumb = (int(lm[4].x * w), int(lm[4].y * h))
        index = (int(lm[8].x * w), int(lm[8].y * h))

        # draw them
        cv2.circle(img, thumb, 8, (0,255,0), cv2.FILLED)
        cv2.circle(img, index, 8, (0,255,0), cv2.FILLED)
        cv2.line(img, thumb, index, (0,255,0), 2)

        # compute distance
        dist = math.hypot(index[0] - thumb[0], index[1] - thumb[1])

        # map to 0–100
        bri = int((dist / MAX_PINCH_DIST) * 100)
        bri = max(0, min(100, bri))

        # rate‐limit to every 0.2 s
        now = time.time()
        if now - last_update > 0.2:
            last_update = now
            brightness = bri
            set_brightness(brightness)

        # show value
        cv2.putText(img, f"{brightness}%", 
                    (thumb[0], thumb[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # display
    cv2.imshow("Pinch to Brightness", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
