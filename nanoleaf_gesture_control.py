import cv2
import mediapipe as mp
import requests
import time

# 1. Nanoleaf setup
NANO_IP    = "192.168.1.42"                # ← replace with your Nanoleaf’s IP
AUTH_TOKEN = "YOUR_AUTH_TOKEN_HERE"        # ← get this from your Nanoleaf’s developer settings
API_BASE   = "http://{NANO_IP}:16021/api/v1/{AUTH_TOKEN}/state"

# 2. Helper functions
def set_brightness(bri: int):
    """Clamp 0–100 and send to Nanoleaf."""
    bri = max(0, min(100, bri))
    payload = {"brightness": bri}
    requests.put(API_BASE, json=payload)

def set_hue(hue: int):
    """Clamp 0–360 and send hue (degree) to Nanoleaf."""
    hue = hue % 360
    payload = {"hue": hue}
    requests.put(API_BASE, json=payload)

# 3. Hand‐tracking init
mp_hands   = mp.solutions.hands
hands      = mp_hands.Hands(min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)
mp_draw    = mp.solutions.drawing_utils
cap        = cv2.VideoCapture(0)

# 4. State variables
prev_center = None
brightness  = 50    # start at mid‐level
hue         = 180   # start at cyan
set_brightness(brightness)
set_hue(hue)
last_update = time.time()

# 5. Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # flip & convert for MediaPipe
    frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results   = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # take first hand
        lm = results.multi_hand_landmarks[0]
        # draw for debugging
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # compute hand centroid as average of landmarks
        pts = [(p.x * frame.shape[1], p.y * frame.shape[0]) for p in lm.landmark]
        cx = sum(x for x, y in pts) / len(pts)
        cy = sum(y for x, y in pts) / len(pts)
        center = (cx, cy)

        if prev_center:
            dx = center[0] - prev_center[0]
            dy = center[1] - prev_center[1]
            mag = (dx**2 + dy**2)**0.5

            # only update every 0.5s to avoid flooding the API
            if mag > 40 and time.time() - last_update > 0.5:
                last_update = time.time()

                # vertical swipe changes brightness
                if abs(dy) > abs(dx):
                    step = int((abs(dy) / mag) * 10)  # proportional step
                    if dy < 0:
                        brightness += step
                    else:
                        brightness -= step
                    set_brightness(brightness)
                    print(f"Brightness → {brightness}")

                # horizontal swipe changes hue
                else:
                    step = int((abs(dx) / mag) * 30)
                    if dx > 0:
                        hue += step
                    else:
                        hue -= step
                    set_hue(hue)
                    print(f"Hue → {hue % 360}")

        prev_center = center

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
