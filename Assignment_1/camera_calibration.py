import numpy as np
import cv2 
import glob
# cap = cv2.VideoCapture(0)

# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# print("height : ", height)
# print("width: ", width)

# #true hahahaha
# i = 0
# while(i < 10):
#     ret, img = cap.read()
#     if(ret):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         cv2.imshow("webcam", gray)
#         foundCorners, corners = cv2.findChessboardCorners(gray, (7,6), None)
#         if(foundCorners):
#             cv2.imwrite("Assignment_1/images/picture_"+ str(i) +".png", gray)
#             cv2.waitKey(3000)
#             i+=1
#     cv2.waitKey(10)
    
# cap.release()
# cv2.destroyAllWindows()

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.png')
print(images)
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
cv2.destroyAllWindows()