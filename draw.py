import cv2
import mediapipe as mp
import numpy as np
import time

# Webcam
cap = cv2.VideoCapture(0)

# Canvas
canvas = np.zeros((720,1280,3),dtype=np.uint8)

prev_x, prev_y = 0,0

# Current drawing color
color = (0,0,255)

# Colors
colors = {
    "red":(0,0,255),
    "green":(0,255,0),
    "blue":(255,0,0),
    "yellow":(0,255,255),
    "white":(255,255,255)
}

# Mediapipe setup
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

    result = detector.detect_for_video(mp_image,timestamp)

    # Draw color palette
    cv2.rectangle(frame,(0,0),(1280,60),(50,50,50),-1)

    cv2.circle(frame,(100,30),20,(0,0,255),-1)
    cv2.circle(frame,(200,30),20,(0,255,0),-1)
    cv2.circle(frame,(300,30),20,(255,0,0),-1)
    cv2.circle(frame,(400,30),20,(0,255,255),-1)
    cv2.circle(frame,(500,30),20,(255,255,255),-1)

    cv2.putText(frame,"CLEAR",(600,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            h,w,_ = frame.shape

            x = int(hand[8].x * w)
            y = int(hand[8].y * h)

            index_up = hand[8].y < hand[6].y
            middle_up = hand[12].y < hand[10].y

            # Color selection
            if y < 60:

                if 80 < x < 120:
                    color = colors["red"]

                elif 180 < x < 220:
                    color = colors["green"]

                elif 280 < x < 320:
                    color = colors["blue"]

                elif 380 < x < 420:
                    color = colors["yellow"]

                elif 480 < x < 520:
                    color = colors["white"]

                elif 560 < x < 700:
                    canvas[:] = 0

                prev_x,prev_y = 0,0

            # Draw mode (index finger only)
            elif index_up and not middle_up:

                if prev_x == 0 and prev_y == 0:
                    prev_x,prev_y = x,y

                cv2.line(canvas,(prev_x,prev_y),(x,y),color,8)

                prev_x,prev_y = x,y

            # Eraser mode (two fingers)
            elif index_up and middle_up:

                cv2.circle(canvas,(x,y),40,(0,0,0),-1)

                prev_x,prev_y = 0,0

            else:
                prev_x,prev_y = 0,0

    output = cv2.addWeighted(frame,0.7,canvas,1.0,0)

    cv2.imshow("AI Gesture Whiteboard",output)

    key = cv2.waitKey(1)

    # Save drawing
    if key == ord('s'):
        cv2.imwrite("drawing.png",canvas)

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()