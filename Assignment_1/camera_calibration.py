import numpy as np
import cv2 
import glob

chessboardWidth = 4
chessboardHeight = 5
maxImageCount = 3

def capturePictures():
    cap = cv2.VideoCapture(0)

    # get camera image parameters from get()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    format = int(cap.get(cv2.CAP_PROP_FORMAT))
    autofocus = int(cap.get(cv2.CAP_PROP_AUTOFOCUS))
    zoom = int(cap.get(cv2.CAP_PROP_ZOOM))
    focus = int(cap.get(cv2.CAP_PROP_FOCUS))

    print('Video properties:')
    print('  Width = ' + str(width))
    print('  Height = ' + str(height))
    print('  Format = ' + str(format))
    print('  Autofocus = ' + str(autofocus))
    print('  Zoom = ' + str(zoom))
    print('  Focus = ' + str(focus))

    if(autofocus == -1):
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off 

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
            cv2.waitKey(2000)
    cv2.destroyAllWindows()
    return objpoints, imgpoints, gray

# capturePictures()
# objpoints, imgpoints, gray = findChessboardCorners()
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# img = cv2.imread('Assignment_1/images/picture_9.png')
# h,  w = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# # undistort
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('Assignment_1/images/calibresult.png', dst)