import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.hands as mpHands
import mediapipe.python.solutions.drawing_utils as mpDraw
import serial
import math


arduino = serial.Serial()

cap=cv2.VideoCapture(0)
hands = mpHands.Hands()

while True:
    ret,detimg=cap.read()
    if not ret:
        print("**Failed to read frame from the camera**")
        break
    detimg = cv2.flip(cv2.resize(detimg,(400,300)), 1) #設定螢幕大小
    if ret:
        imgRGB = cv2.cvtColor(detimg,cv2.COLOR_BGR2RGB) 
        resule = hands.process(imgRGB)
        if resule.multi_hand_landmarks:
                for handLms in resule.multi_hand_landmarks:
                    mpDraw.draw_landmarks(detimg,handLms,mpHands.HAND_CONNECTIONS)
                    for i,lm in enumerate(handLms.landmark):
                        print(i , int(lm.x*detimg.shape[0]) , int(lm.y*detimg.shape[1]))
                thumb_base = handLms.landmark[mpHands.HandLandmark.THUMB_MCP]
                thumb_tip = handLms.landmark[mpHands.HandLandmark.THUMB_TIP]
                thumb_tip_x, thumb_tip_y = int(thumb_tip.x * detimg.shape[1]), int(thumb_tip.y * detimg.shape[0])
                thumb_base_x,thumb_base_y = int(thumb_base.x *detimg.shape[1]),int(thumb_base.y * detimg.shape[0])
                indexfinger_tip = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
                middlefinger_tip = handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP]
                indexfinger_tip_x,indexfinger_tip_y = int(indexfinger_tip.x*detimg.shape[1]),int(indexfinger_tip.y*detimg.shape[0])
                middlefinger_tip_x,middlefinger_tip_y = int(middlefinger_tip.x*detimg.shape[1]),int(middlefinger_tip.y*detimg.shape[0])
                def slope(xt,xb,yt,yb):
                    m = (xt-xb)/(yt-yb)
                    return int(m)
                #拇指tip&base的斜率要再設置過
            
                if thumb_tip_x - thumb_base_x < 0 and slope(thumb_tip_x,thumb_base_x,thumb_tip_y,thumb_base_y)<(0):
                         cv2.circle(detimg, (200, 100), 5, (0, 0, 255), -1)
                else:
                     None

                
                

    cv2.imshow('img',detimg)
    key=cv2.waitKey(2)
    if key ==27:
        break
cap.release
cv2.destroyAllWindows