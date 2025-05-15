# main.py
import cv2
import time
import os
import importlib

from nanoleaf_utils import set_brightness

# initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# load gesture module
gesture = importlib.import_module("gesture")
last_mtime = os.path.getmtime("gesture.py")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # hot-reload gesture.py if changed
    mtime = os.path.getmtime("gesture.py")
    if mtime != last_mtime:
        gesture = importlib.reload(gesture)
        last_mtime = mtime
        print("🔄 gesture.py reloaded")

    # process frame → get target brightness + annotated frame
    bri, vis = gesture.process_frame(frame)

    # send to Nanoleaf
    set_brightness(bri)

    cv2.imshow("Live Gesture Control", vis)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
