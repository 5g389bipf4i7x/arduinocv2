import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.hands as mpHands
import mediapipe.python.solutions.drawing_utils as mpDraw
import math

cap = cv2.VideoCapture(0)
hands = mpHands.Hands()

def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_ = math.degrees(math.acos((v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
    except:
        angle_ = 180
    return angle_

def hand_angle(hand_):
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
        ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
    )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
        ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
    )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
        ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
    )
    angle_list.append(angle_)
    # ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
        ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
    )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
        ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
    )
    angle_list.append(angle_)
    return angle_list

def hand_pos(finger_angle):
    f1 = finger_angle[0]  # 大拇指角度
    f2 = finger_angle[1]  # 食指角度
    f3 = finger_angle[2]  # 中指角度
    f4 = finger_angle[3]  # 無名指角度
    f5 = finger_angle[4]  # 小拇指角度
    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
    if f1 < 50 and f3 > 50:
        return 'catch'
    return 'open'

# 初始化狀態變量
catch_count = 0
distance_tracking = False

while True:
    ret, detimg = cap.read()
    if not ret:
        print("**Failed to read frame from the camera**")
        break

    detimg = cv2.flip(cv2.resize(detimg, (400, 300)), 1)  # 設定螢幕大小
    imgRGB = cv2.cvtColor(detimg, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    Cheight, Clength = detimg.shape[:2]
    centerpoint = (Clength // 2, Cheight // 2)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(detimg, handLms, mpHands.HAND_CONNECTIONS)
            finger_points = []
            for lm in handLms.landmark:  # 取得座標
                finger_x = lm.x * detimg.shape[1]
                finger_y = lm.y * detimg.shape[0]
                finger_points.append((finger_x, finger_y))

            angles = hand_angle(finger_points)
            hand_state = hand_pos(angles)

            if hand_state == 'catch':
                catch_count += 1
                if catch_count == 1:
                    print("Start tracking distance...")
                    distance_tracking = True
                elif catch_count == 2:
                    print("Stop tracking distance.")
                    distance_tracking = False
                    catch_count = 0  # 重置計數器

            if distance_tracking:
                distance_point = [abs(finger_points[12][0] - finger_points[4][0]), abs(finger_points[12][1] - finger_points[4][1])]  # 取座標相減絕對值
                distance = math.sqrt(distance_point[0] ** 2 + distance_point[1] ** 2)  # 取得拇指與中指之距
                print(f"Distance between thumb and middle finger: {distance}")

    cv2.imshow("Frame", detimg)
    if cv2.waitKey(1) & 0xFF == 27:  # 按 Esc 鍵退出
        break

cap.release()
cv2.destroyAllWindows()
