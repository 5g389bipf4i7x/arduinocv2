import cv2
import numpy as np
import mediapipe.python.solutions.hands as mpHands
import math
import pyfirmata2
import time

board = pyfirmata2.Arduino('COM3') #初始化uno板及伺服馬達
time.sleep(1)
servos = {
    'claw' : board.get_pin('d:6:s'),
    'base' : board.get_pin('d:11:s'),
    'right' : board.get_pin('d:9:s'),
    'left' : board.get_pin('d:10:s')
}
time.sleep(1)
servos['base'].write(90) #馬達初始定位
servos['left'].write(70)
servos['right'].write(90)

def command(leftmiddle_x,leftmiddle_y,leftmiddle_z): #手勢控制(上下左右前後)
    if (9*leftmiddle_x-11*leftmiddle_y>0) and (11*leftmiddle_x+9*leftmiddle_y<8080) and leftmiddle_y<250:
        common = 'UP'
    elif (9*leftmiddle_x-11*leftmiddle_y<0) and (11*leftmiddle_x+9*leftmiddle_y>8080) and leftmiddle_y>470:
        common = 'DOWN'
    elif (9*leftmiddle_x-11*leftmiddle_y<0) and (11*leftmiddle_x+9*leftmiddle_y<8080) and leftmiddle_x<330:
        common = 'LEFT'
    elif (9*leftmiddle_x-11*leftmiddle_y>0) and (11*leftmiddle_x+9*leftmiddle_y>8080) and 500 <leftmiddle_x <880:
        common = 'RIGHT'
    elif 330 <=leftmiddle_x <=550 and 250 <= leftmiddle_y <=470 and leftmiddle_z<-0.08:
        common = 'FRONT'
    elif 330 <=leftmiddle_x <=550 and 250 <= leftmiddle_y <=470 and leftmiddle_z>-0.03:
        common = 'BACK'
    else:
        common = 'STAY'
    return common

#繪製直線
def draw_arrowedLIne(detimg,arrowed): 
    for start,end in arrowed:
        cv2.arrowedLine
        (detimg,start,end,(225,225,225),2)

def draw_Line(detimg,line):
    for start,end in line:
        cv2.line
        (detimg,start,end,(225,225,225))

pos_base = 90
pos_left = 90
pos_right = 90

cap=cv2.VideoCapture(0) #初始化鏡頭
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

while True:
    ret,detimg=cap.read()
    if not ret:
        print("**Failed to read frame from the camera**")
        break
    else:
        detimg = cv2.flip(cv2.resize(detimg,(1280,720)), 1) #設定螢幕大小
        imgRGB = cv2.cvtColor(detimg,cv2.COLOR_BGR2RGB)  #轉換影像顏色格式
        results = hands.process(imgRGB)
        cv2.line(detimg,(880,0),(880,720),(255,255,255),2)
        cv2.circle(detimg,(440,360), 5, (255,225,225),-1)
        arrowedline = [
            ((440,250),(440,180)),
            ((440,470),(440,540)),
            ((330,360),(260,360)),
            ((550,360),(620,360))
        ]
        draw_arrowedLIne(detimg,arrowedline) 
        area = [
            ((330,470),(350,470)),
            ((330,470),(330,450)),
            ((550,470),(530,470)),
            ((550,470),(550,450)),
            ((330,250),(350,250)),
            ((330,250),(330,270)),
            ((550,250),(530,250)),
            ((550,250),(550,270))
        ]
        draw_Line(detimg,area)
        if results.multi_hand_landmarks:
            for handLms,handedness in zip(results.multi_hand_landmarks, results.multi_handedness):                    
                hand_type = handedness.classification[0].label #辨識左右手
                if hand_type == 'Right':
                    right_thumb = handLms.landmark[mpHands.HandLandmark.THUMB_TIP]
                    right_middle = handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP]
                    right_thumb_x = right_thumb.x*detimg.shape[1]
                    right_thumb_y = right_thumb.y*detimg.shape[0]
                    right_middle_x = right_middle.x*detimg.shape[1]
                    right_middle_y = right_middle.y*detimg.shape[0]
                    distance = round(math.sqrt((right_middle_x - right_thumb_x)**2 + (right_middle_y - right_thumb_y)**2)) #運算兩指間距
                    distance = np.clip(distance, 0, 220) #將距離映射於0-220之間
                    pos_claw=(145-np.interp(distance,[0,220],[0,145])) 
                    servos['claw'].write(pos_claw) #驅動爪子伺服馬達
                    time.sleep(0.2)
                if hand_type == 'Left':
                    left_middle = handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP]
                    left_middle_x = left_middle.x*detimg.shape[1]
                    left_middle_y = left_middle.y*detimg.shape[0]
                    signal = command(int(left_middle_x),int(left_middle_y),float(left_middle.z)) 
                    print(signal)
                    if signal == 'LEFT' and pos_base >0:
                        pos_base -= 1
                    elif signal == 'RIGHT' and pos_base <180:
                        pos_base += 1
                    elif signal == 'DOWN' and pos_left>0:
                        pos_left -= 1
                    elif signal == 'UP' and pos_left<90: 
                        pos_left += 1
                    elif signal == 'FRONT' and pos_right<170:
                        pos_right += 1
                    elif signal == 'BACK' and pos_right>90:
                        pos_right -= 1
                    elif signal == 'STAY':
                        pass

                    if signal == 'LEFT' or signal == 'RIGHT':
                        servos['base'].write(pos_base)
                        time.sleep(0.1)
                    elif signal == 'UP' or signal == 'DOWN':
                        servos['left'].write(pos_left)
                        time.sleep(0.1)
                    elif signal == 'FRONT' or signal == 'BACK':
                        servos['right'].write(pos_right)
                        time.sleep(0.1)
                    elif signal == 'STAY':
                        pass

    cv2.imshow('img',detimg)
    if cv2.waitKey(5) ==27: #按下Esc鍵即退出
        break


cap.release
cv2.destroyAllWindows
board.exit()
