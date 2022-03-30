import numpy as np
import cv2 
import glob

chessboardWidth = 9
chessboardHeight = 6
maxImageCount = 10

def capturePictures():
    cap = cv2.VideoCapture(0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    i = 0
    while(i < maxImageCount):
        ret, img = cap.read()
        if(ret):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow("webcam", gray)
            foundCorners, corners = cv2.findChessboardCorners(gray, (chessboardWidth,chessboardHeight), None)
            if(foundCorners):
                cv2.imwrite("Assignment_1/images/picture_"+ str(i) +".png", gray)
                cv2.waitKey(5000)
                i+=1
        cv2.waitKey(1)
        
    cap.release()
    cv2.destroyAllWindows()

def findChessboardCorners():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardHeight*chessboardWidth,3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardWidth,0:chessboardHeight].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('Assignment_1/images/*.png')
    print(images)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (chessboardWidth,chessboardHeight), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (chessboardWidth,chessboardHeight), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(5000)
    cv2.destroyAllWindows()

findChessboardCorners()