import numpy as np
import cv2 

cap = cv2.VideoCapture(0)

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print("height : ", height)
print("width: ", width)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = [] # 3d point in real world space

#true hahahaha
while(True):
    i = 0
    ret, img = cap.read()
    if(ret):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("webcam", gray)
        foundCorners, corners = cv2.findChessboardCorners(gray, (7,6), None)
        if(foundCorners):
            cv2.imwrite("images/picture_"+ str(i) +".png", gray)
            i+=1
            break
    if cv2.waitKey(10) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

