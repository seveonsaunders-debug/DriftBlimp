# a simple test of cv2 with yolo

import cv2
from ultralytics import YOLO

cv2.namedWindow("preview")

vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        cv2.destroyWindow("preview")
        vc.release()
        break
    elif key == ord('c'):  # capture image
        cv2.imshow("captured",frame) # shows captured image in new window
        if key == ord('s'): # save image
            cv2.imwrite("captured_image.jpg", frame)
            print("Image saved")
            break


cv2.destroyWindow("preview")
vc.release()
