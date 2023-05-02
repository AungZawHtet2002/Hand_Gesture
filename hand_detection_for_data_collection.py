import mediapipe as mp
import cv2 
hands=mp.solutions.hands
drawing=mp.solutions.drawing_utils

webcam= cv2.VideoCapture(0)
while webcam.isOpened():
    success,img= webcam.read()
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=hands.Hands().process(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            drawing.draw_landmarks(img,landmarks,connections=hands.HAND_CONNECTIONS)

    cv2.imshow('VD',img)
    if cv2.waitKey(0) & 0xFF == ord ('q'):
        break

 