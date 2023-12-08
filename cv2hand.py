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
    ret,detimg=cap.read()
    detimg = cv2.flip(cv2.resize(detimg,(400,300)), 1)
    if ret:
        imgRGB = cv2.cvtColor(detimg,cv2.COLOR_BGR2RGB)
        resule = hands.process(imgRGB)
        if resule.multi_hand_landmarks:
                for handLms in resule.multi_hand_landmarks:
                    mpDraw.draw_landmarks(detimg,handLms,mpHands.HAND_CONNECTIONS)
                    for i,lm in enumerate(handLms.landmark):
                        print(i , int(lm.x*detimg.shape[0]) , int(lm.y*detimg.shape[1]))

        imgray = cv2.cvtColor(detimg,cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(imgray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        contours,_ =cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt)> min_area 
                             and cv2.contourArea(cnt)<max_area] #手部面積尚未被定義

        cv2.imshow('Hand Contour Detection',detimg)
    
    key=cv2.waitKey(2)
    if key ==27:
        break
cap.release
cv2.destroyAllWindows