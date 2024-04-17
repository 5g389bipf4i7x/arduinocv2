import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.hands as mpHands
import mediapipe.python.solutions.drawing_utils as mpDraw
import serial
import math
import sys

cap=cv2.VideoCapture(0)
hands = mpHands.Hands()
#arduino = serial.Serial()

while True:
    ret,detimg=cap.read()
    if not ret:
        print("**Failed to read frame from the camera**")
        break
    else:
        detimg = cv2.flip(cv2.resize(detimg,(400,300)), 1) #設定螢幕大小
        imgRGB = cv2.cvtColor(detimg,cv2.COLOR_BGR2RGB)  
        resule = hands.process(imgRGB)
        lm12x = 0 
        lm12y = 0  
        lm4x = 0  
        lm4y = 0
        Cheight,Clength = detimg.shape[:2]
        centerpoint= (Clength//2,Cheight//2)
        if resule.multi_hand_landmarks:
            for handLms in resule.multi_hand_landmarks:
                mpDraw.draw_landmarks(detimg,handLms,mpHands.HAND_CONNECTIONS)
                for lm in enumerate(handLms.landmark): #取得座標
                    #print(i , int(lm.x*detimg.shape[1]) , int(lm.y*detimg.shape[0]),int(lm.z*detimg.shape[2])) 
                    lm12x = int(handLms.landmark[12].x*detimg.shape[1]) #取得12節點的X座標
                    lm12y = int(handLms.landmark[12].y*detimg.shape[0]) #...Y座標
                    cv2.circle ( detimg,(lm12x,lm12y) , 5 , (0,0,255) , cv2.FILLED) #在中指尖上畫圓
                    lm4x = int(handLms.landmark[4].x*detimg.shape[1]) #取得4節點的X座標
                    lm4y = int(handLms.landmark[4].y*detimg.shape[0])
        cv2.circle (detimg,(centerpoint),10,(255,0,0),cv2.FILLED) #中心定一點
        if lm4x ==  lm12x and lm4y == lm12y:
            while True:
                distancepoint = [abs(lm12x-lm4x),abs(lm12y-lm4y)]
                distance = math.sqrt(distancepoint[0]**2 + distancepoint[1]**2) #取得拇指與中指之距
                print(distance)# 會因為靠近鏡頭而讓值變大 暫時無法修正 待量化距離&手臂

                if lm4x ==  lm12x and lm4y == lm12y:
                    break
#拇指與中指併起 食指靠攏 抓握啟動 再靠攏一次 抓握型態結束 當進入抓握狀態時無法移動 或是以pose偵測節點

        #底盤旋轉 將畫面分成四等分 取第2第3部分作偵測範圍

        #y軸與pose判斷手臂抬升下降 手腕掌跟-小臂 手軸掌跟-大臂        
        



                

    cv2.imshow('img',detimg)
    key=cv2.waitKey(2)
    if key ==27:
        break
cap.release
cv2.destroyAllWindows