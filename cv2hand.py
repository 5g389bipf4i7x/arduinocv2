import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.hands as mpHands
import mediapipe.python.solutions.drawing_utils as mpDraw
import serial


arduino = serial.Serial()

cap=cv2.VideoCapture(0)
hands = mpHands.Hands()
# min_tracking_confidence=0.5 追蹤狀態 數值低刷新慢
while True:
    ret,img=cap.read()
    img = cv2.flip(cv2.resize(img,(400,300)), 1)
    if ret:
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        resule = hands.process(imgRGB)
        # print(resule.multi_hand_landmarks)
        if resule.multi_hand_landmarks:
            for handLms in resule.multi_hand_landmarks:
                mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
                for i,lm in enumerate(handLms.landmark):
                    print(i , int(lm.x*img.shape[0]) , int(lm.y*img.shape[1]))

        cv2.imshow('img',img)
    
    key=cv2.waitKey(2)
    if key ==27:
        break