import cv2
import mediapipe as mp
import os
import time
from gesture import Hand

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HERE = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(HERE, "gesture_recognizer.task")

# Create a gesture recognizer instance with the video mode:
try:
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    gestureRecognizer = GestureRecognizer.create_from_options(options)
except Exception as e:
    raise IOError(f"Failed to load gesture recognizer model. Please download it from https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task and place it in the same directory as the script. Error: {e}")


_last_time = time.time()

        
def process_frame(frame) -> tuple[Hand, float]:

    global _last_time

    img = cv2.flip(frame, 1)
    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    ts_ms = int(time.time() * 1000)
    gesture_recognition_result = gestureRecognizer.recognize_for_video(mp_image, ts_ms)

    

        
    if gesture_recognition_result.handedness:
        handedness = gesture_recognition_result.handedness[0][0]
        confidence = handedness.score
        if handedness.category_name == "Left":
            return Hand.RIGHT, confidence
        elif handedness.category_name == "Right":
            return Hand.LEFT, confidence
        
    return None, 0.0