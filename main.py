# main.py
import cv2
import time
import os
import importlib
import gesture_recognizer
from nanoleaf_utils import get_brightness, set_brightness
import face_orientation
from gesture import Hand


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

    hand, confidence = gesture_recognizer.process_frame(frame)

    happening, conf, pose, vis2 = face_orientation.process_frame(frame)


    # send to Nanoleaf
    if bri >= 0 and hand == Hand.RIGHT and happening:
        set_brightness(bri)
        print(f"Hand: {hand}, Confidence: {confidence:.2f}, Setting brightness to {bri:.1f}%")
    else:
        last_brightness = get_brightness()
        set_brightness(last_brightness if 'last_brightness' in locals() else 0)

        print("Pinky down and - brightness not updated")

    # display

    # Show the face-based visualization instead of (or alongside) your existing vis
    cv2.imshow("Face Orientation", vis2)


    

    cv2.imshow("Live Gesture Control", vis)
    cv2.waitKey(1)

    # if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
    #     break

    time.sleep(0.01)  # small delay to reduce CPU usage

cap.release()
cv2.destroyAllWindows()
