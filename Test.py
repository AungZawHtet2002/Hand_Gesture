import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf 

detector= HandDetector(maxHands=1)
offset=30                    
imgsize=256
class_name=['Go','Left','Right']
model=tf.keras.models.load_model("hand_model1.h5")   #load our model
url = "http://192.168.100.9:8080/video"
cap = cv2.VideoCapture(url)
previous= "G"
while True:
    success,img=cap.read()
    Output= img.copy()
    hands,img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        whiteimg=np.ones((imgsize,imgsize,3),np.uint8)*255
        croped_img=img[ y-offset:y+h+offset , x-offset:x+w+offset ]
        
        ratio= h/w
        if ratio > 1 :
            constant=imgsize/h
            new_w= math.ceil(constant * w)
            resize_img=cv2.resize(croped_img,(new_w,imgsize))
            w_gap= math.ceil((imgsize-new_w)/2)
            whiteimg[:,w_gap:new_w+w_gap]=resize_img

        else:
            constant=imgsize/w
            new_h= math.ceil(constant * h)
            resize_img=cv2.resize(croped_img,(imgsize,new_h))
            h_gap= math.ceil((imgsize-new_h)/2)
            whiteimg[h_gap:new_h+h_gap,:]=resize_img
        cv2.imshow("Croped Image",whiteimg)
    data = np.expand_dims(whiteimg, axis=0)
    predictions=model.predict(data)
    predicted_class=class_name[np.argmax(predictions[0])]
    cv2.putText(Output,predicted_class,(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),2)
    cv2.rectangle(Output,(x-offset,y-offset),(x+w+offset,y+h+offset),(0,255,0),4)
    cv2.imshow("Video",Output)
    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break
cap.release()
cv2.destroyAllWindows()