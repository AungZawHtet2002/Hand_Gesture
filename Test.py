import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
cap=cv2.VideoCapture(0)
detector= HandDetector(maxHands=1)
offset=30                    
imgsize=256

model=tf.keras.models.load_model("model")   #load our model
cap= cv2.VideoCapture(0)
class_name=['Go','Left','Right']
while True:
    success,img=cap.read()
    hands,img =  detector.findHands(img)
    if hands:
        hand=hand[0]
        x,y,w,h=hand['bbox']
        whiteimg=np.ones((imgsize,imgsize,3),np.uint8)
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
    image = np.expand_dims(whiteimg, axis=0)
    predictions=model.predict(image)
    predicted_class=class_name[np.argmax(predictions[0])]
    print(predicted_class)
    cv2.imshow("Video",img)
    if cv2.waitKey(0) & 0xFF == ord ('q'):
        break

cap.release()
cv2.destroyAllWindows()