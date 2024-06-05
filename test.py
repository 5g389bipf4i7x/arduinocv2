import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.hands as mpHands
import mediapipe.python.solutions.drawing_utils as mpDraw


cap=cv2.VideoCapture(0)
hands = mpHands.Hands()

while True:
    ret,detimg=cap.read()
    if not ret:
        print("**Failed to read frame from the camera**")
        break
    else:
        detimg = cv2.flip(cv2.resize(detimg,(400,300)), 1) #設定螢幕大小
        imgRGB = cv2.cvtColor(detimg,cv2.COLOR_BGR2RGB)  
        resule = hands.process(imgRGB)
        Cheight,Clength = detimg.shape[:2]
        centerpoint= (Clength//2,Cheight//2)
        if resule.multi_hand_landmarks:
            for handLms in resule.multi_hand_landmarks:
                mpDraw.draw_landmarks(detimg,handLms,mpHands.HAND_CONNECTIONS)
                finger_points=[]
                for lm in handLms.landmark: #取得座標
                    finger_x = lm.x*detimg.shape[1]
                    finger_y = lm.y*detimg.shape[0]
                    finger_points.append((finger_x,finger_y))
                    print(finger_x,finger_y)
                for i,lms in enumerate(handLms.landmark):
                    print(i,lms.x,lms.y)


    cv2.imshow('img',detimg)
    key=cv2.waitKey(2)
    if key ==27:
        break
cap.release
cv2.destroyAllWindows