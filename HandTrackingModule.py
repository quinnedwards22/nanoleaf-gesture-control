import cv2
import mediapipe as mp
import time
import numpy as np

# STEP 1: Create an HandLandmarker object.
model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

detector = HandLandmarker.create_from_options(options)

# Define HAND_CONNECTIONS
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    
    detection_result = detector.detect_for_video(mp_image, int(time.time() * 1000))

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw landmarks
            for id, lm in enumerate(hand_landmarks):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)

            # Draw connections
            for connection in HAND_CONNECTIONS:
                x1, y1 = hand_landmarks[connection[0]].x, hand_landmarks[connection[0]].y
                x2, y2 = hand_landmarks[connection[1]].x, hand_landmarks[connection[1]].y
                h, w, c = img.shape
                cv2.line(img, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0,255,0), 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
