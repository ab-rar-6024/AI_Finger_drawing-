import cv2
import mediapipe as mp
import numpy as np
import time

cap = cv2.VideoCapture(0)

canvas = np.zeros((720,1280,3),dtype=np.uint8)

prev_x, prev_y = 0,0

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

timestamp = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,(1280,720))

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

    timestamp += 1

    result = detector.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            h,w,_ = frame.shape

            x = int(hand[8].x * w)
            y = int(hand[8].y * h)

            if prev_x == 0 and prev_y == 0:
                prev_x,prev_y = x,y

            cv2.line(canvas,(prev_x,prev_y),(x,y),(0,0,255),8)

            prev_x,prev_y = x,y

    else:
        prev_x,prev_y = 0,0

    output = cv2.addWeighted(frame,0.7,canvas,1.0,0)

    cv2.imshow("AI Finger Drawing",output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
