import cv2
import mediapipe as mp
import math
import numpy as np
import tensorflow as tf
# Read the image
#image = cv2.imread('1.jpg')
url = "http://192.168.100.9:8080/video"
cp = cv2.VideoCapture(url)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
imgsize=256
offset = 40
class_name=['Go','Left','Right']
model=tf.keras.models.load_model("hand_model2.h5")   #load our model
def get_bbox(image):
    for hand_landmarks in results.multi_hand_landmarks:
        # Get the bounding box coordinates of the hand
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
                w=x_max-x_min
                h=y_max-y_min
    return x_min,y_min,w,h
# hands=mp.hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5)
# # Resize the image to a specific width and height
#resized_image = cv2.resize(image, (256, 256))
while(True):
    camera, image = cp.read()
    with mp_hands.Hands(min_detection_confidence=0.4,min_tracking_confidence=0.5) as hands:
        results=hands.process(image)
        if results.multi_hand_landmarks:
            for num,hand in enumerate (results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS)
            x,y,w,h=get_bbox(image)
            croped_img=image[ y-offset:y+h+offset , x-offset:x+w+offset ]
            whiteimg=np.ones((imgsize,imgsize,3),np.uint8)
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
        image = np.expand_dims(whiteimg, axis=0)
        predictions=model.predict(image)
        predicted_class=class_name[np.argmax(predictions[0])]
    if predicted_class!= previous:
            previous=predicted_class
            print(predicted_class)
            cv2.imshow("Video", whiteimg)
    q = cv2.waitKey(1)
    if q==ord("q"):
        break
cp.release()
cv2.destroyAllWindows()
        


