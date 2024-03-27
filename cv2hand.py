import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.hands as mpHands
import mediapipe.python.solutions.drawing_utils as mpDraw
import serial
import math


# arduino = serial.Serial()

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
        if resule.multi_hand_landmarks:
                for handLms in resule.multi_hand_landmarks:
                    mpDraw.draw_landmarks(detimg,handLms,mpHands.HAND_CONNECTIONS)
                    for i,lm in enumerate(handLms.landmark):
                        #print(i , int(lm.x*detimg.shape[0]) , int(lm.y*detimg.shape[1])) #
                        lm12x = int(handLms.landmark[12].x*detimg.shape[1]) #取得12節點的X座標
                        lm12y = int(handLms.landmark[12].y*detimg.shape[0]) #...Y座標
                        cv2.circle ( detimg,(lm12x,lm12y) , 10 , (0,0,255) , cv2.FILLED) #在12上畫圓
                    
                length,height,_ = detimg.shape
                cameracenter= length//2,height//2

                #要分割攝影機畫面成9等分或26等分-放射
                
                #抓握



                

    cv2.imshow('img',detimg)
    key=cv2.waitKey(2)
    if key ==27:
        break
cap.release
cv2.destroyAllWindows