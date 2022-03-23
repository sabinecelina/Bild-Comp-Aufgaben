import numpy as np
import cv2 

cap = cv2.VideoCapture(0)

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print("height : ", height)
print("width: ", width)

while(foundCorners):
    ret, img = cap.read()
    if(ret):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("webcam", gray)
        foundCorners, corners = cv2.findChessboardCorners(gray, (7,6), None)


cap.release()
cv2.destroyAllWindows()

