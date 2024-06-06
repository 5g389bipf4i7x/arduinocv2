import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.hands as mpHands
import mediapipe.python.solutions.drawing_utils as mpDraw
import math
import sys
import pyfirmata2

cap=cv2.VideoCapture(0)
hands = mpHands.Hands()

board = pyfirmata2.Arduino()
servo = board.get_pin()


def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 180
    return angle_

def hand_angle(hand_):
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    # ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

def hand_pos(finger_angle):
    f1 = finger_angle[0]   # 大拇指角度
    f2 = finger_angle[1]   # 食指角度
    f3 = finger_angle[2]   # 中指角度
    f4 = finger_angle[3]   # 無名指角度
    f5 = finger_angle[4]   # 小拇指角度
    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮    
    if f1>50 and f2>60  and f3>50 and f4>60 and f5>60:
        return 'catch'

THRESHOLD = 30
def hand_pos_and_control(finger_points, cx, cy):
    # 取中指指根 (9) 位置
    middle_finger_mcp = finger_points[9]
    dx = middle_finger_mcp[0] - cx
    dy = middle_finger_mcp[1] - cy
    
    if abs(dx) <=THRESHOLD and abs(dy) <=THRESHOLD:
        command = 'STAY'
    elif dx > THRESHOLD and abs(dy) <= THRESHOLD:
        command = 'RIGHT'
    elif dx < -THRESHOLD and abs(dy) <= THRESHOLD:
        command = 'LEFT'
    elif abs(dx) <= THRESHOLD and dy > THRESHOLD:
        command = 'DOWN'
    elif abs(dx) <= THRESHOLD and dy < -THRESHOLD:
        command = 'UP'
    elif dx > THRESHOLD and dy > THRESHOLD:
        command = 'RIGHT,DOWN'
    elif dx > THRESHOLD and dy < -THRESHOLD:
        command = 'RIGHT,UP'
    elif dx < -THRESHOLD and dy > THRESHOLD:
        command = 'LEFT,DOWN'
    elif dx < -THRESHOLD and dy < -THRESHOLD:
        command = 'LEFT,UP'
    else:
        command = 'STAY'

    return command


#轉手
catch_count = 0
distance_tm = False

while True:
    ret,detimg=cap.read()
    if not ret:
        print("**Failed to read frame from the camera**")
        break
    else:
        detimg = cv2.flip(cv2.resize(detimg,(640,480)), 1) #設定螢幕大小
        imgRGB = cv2.cvtColor(detimg,cv2.COLOR_BGR2RGB)  
        resule = hands.process(imgRGB)
        Cheight,Clength = detimg.shape[:2]
        cx,cy= (Clength//2,Cheight//2)
        cv2.circle(detimg,(cx, cy), 5, (255, 0, 0), -1)
        if resule.multi_hand_landmarks:
            for handLms in resule.multi_hand_landmarks:
                mpDraw.draw_landmarks(detimg,handLms,mpHands.HAND_CONNECTIONS)
                finger_points=[]
                for lm in handLms.landmark: #取得座標
                    finger_x = lm.x*detimg.shape[1]
                    finger_y = lm.y*detimg.shape[0]
                    finger_points.append((finger_x,finger_y))
                
                angles = hand_angle(finger_points)
                if hand_pos(angles) == 'catch':
                    catch_count += 1
                    if catch_count == 1:
                        distance_tm = True
                    elif catch_count == 2:
                        distance_tm = False
                        catch_count = 0
                if distance_tm:
                    distancepoint = [abs(finger_points[12][0]-finger_points[4][0]),abs(finger_points[12][1]-finger_points[4][1])]#取座標相減絕對值
                    distance = math.sqrt(distancepoint[0]**2 + distancepoint[1]**2) #取得拇指與中指之距
                    print(distance)
                if distance_tm == False:
                    command = hand_pos_and_control(finger_points, cx, cy)
                    print(command)
                clip_distance = round(np.interp(distance,[],[0,100]))
                servo_cdt = (100-clip_distance)
                servo.write(servo_cdt)

                
    cv2.imshow('img',detimg)
    key=cv2.waitKey(2)
    if key ==27:
        break


cap.release
cv2.destroyAllWindows
