# a simple test of cv2 with yolo

import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0) # set source camera
ret,frame = cap.read() # captures a single frame

while(True):
    cv2.imshow('img1',frame) # display frame
    if cv2.waitKey(1) & 0xFF == ord('s'): # press 's' to save frame
        cv2.imwrite('img1.jpg',frame) # save frame
        break
cap.release() # release camera


model = yolo('glasses.pt') # load yolo model
