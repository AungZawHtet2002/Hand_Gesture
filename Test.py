import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf 

detector= HandDetector(maxHands=1)
offset=30                    
imgsize=256
class_name=['Go','Left','Right','Stop']
model=tf.keras.models.load_model("66model.h5")   #load our model

cap = cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    Output= img.copy()
    hands,img=detector.findHands(img)       #hand detection on video

    if hands:                               #Hand detection success
        hand=hands[0]
        x,y,w,h=hand['bbox']                #Get bounding box coordinates and size
        whiteimg=np.ones((imgsize,imgsize,3),np.uint8)*255      #Create image boundry to crop hands from video 
        croped_img=img[ y-offset:y+h+offset , x-offset:x+w+offset ]     #Crop Image success
        
        ratio= h/w                             #Find ratio to fix image in image boundry
        if ratio > 1 :
            constant=imgsize/h
            new_w= math.ceil(constant * w)
            resize_img=cv2.resize(croped_img,(new_w,imgsize))       #Resize croped image
            w_gap= math.ceil((imgsize-new_w)/2)
            whiteimg[:,w_gap:new_w+w_gap]=resize_img                #Put croped image in image boundry

        else:
            constant=imgsize/w
            new_h= math.ceil(constant * h)
            resize_img=cv2.resize(croped_img,(imgsize,new_h))
            h_gap= math.ceil((imgsize-new_h)/2)
            whiteimg[h_gap:new_h+h_gap,:]=resize_img

        cv2.imshow("Croped Image",whiteimg)         #Show image boundry that is cropped from video

    data = np.expand_dims(whiteimg, axis=0)
    predictions=model.predict(data)
    predicted_class=class_name[np.argmax(predictions[0])]       #Predict output class
    cv2.putText(Output,predicted_class,(x+40,y-40),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),2)        #Text in  video
    cv2.rectangle(Output,(x-offset,y-offset),(x+w+offset,y+h+offset),(0,255,0),4)                   #Draw bounding box in video

    cv2.imshow("Video",Output)
    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break
cap.release()
cv2.destroyAllWindows()