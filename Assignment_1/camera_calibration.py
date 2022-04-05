import numpy as np
import cv2
import glob

chessboardWidth = 4
chessboardHeight = 5
maxImageCount = 10


def capturePictures():
    cap = cv2.VideoCapture(0)

    # get camera image parameters from get()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    autofocus = int(cap.get(cv2.CAP_PROP_AUTOFOCUS))
    zoom = int(cap.get(cv2.CAP_PROP_ZOOM))
    focus = int(cap.get(cv2.CAP_PROP_FOCUS))

    print('Video properties:')
    print('  Width = ' + str(width))
    print('  Height = ' + str(height))
    print('  Autofocus = ' + str(autofocus))
    print('  Zoom = ' + str(zoom))
    print('  Focus = ' + str(focus))

    if autofocus != -1:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off

    for x in range(maxImageCount):
        ret, img = cap.read()
        if ret:
            # findChessboardCorners needs a grayscale image
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow("webcam", gray_img)
            found_corners, corners = cv2.findChessboardCorners(gray_img, (chessboardWidth, chessboardHeight), None)
            if found_corners:
                cv2.imwrite("Assignment_1/images/picture_" + str(x) + ".png", gray_img)
                cv2.waitKey(5000)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()


# inspiriert von https://docs.opencv.org/4.5.5/dc/dbb/tutorial_py_calibration.html
def find_chessboard_corners():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((chessboardHeight * chessboardWidth, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardWidth, 0:chessboardHeight].T.reshape(-1, 2)
    print(objp)
    obj_points = []
    img_points = []
    images = glob.glob('images/*.png')
    gray_img = 0
    for frame in images:
        img = cv2.imread(frame)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_img, (chessboardWidth, chessboardHeight), None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (chessboardWidth, chessboardHeight), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(2)
    cv2.destroyAllWindows()
    return obj_points, img_points, gray_img


# inspiriert von https://docs.opencv.org/4.5.5/dc/dbb/tutorial_py_calibration.html
def undistorted_live_cam(mtx, dist):
    cap = cv2.VideoCapture(0)
    check = False
    while True:
        # read one video frame
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            dst = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            if cv2.waitKey(10) == ord('u'):
                check = not check
            if check:
                frame = dst
            cv2.imshow('video image', frame)
            if cv2.waitKey(10) == ord('q'):
                break
        else:
            print('Error reading frame')
            break
    cap.release()
    cv2.destroyAllWindows()


# capturePictures()

object_points, image_points, gray = find_chessboard_corners()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
print("camera matrix: \n ", mtx)
print("distortion coefficients: ", dist)
for i in range(len(rvecs)):
    print("\n parameters for image ", i + 1, ": \n")
    print("rotation vector: \n", rvecs[i])
    print("translation vector: \n", tvecs[i])

# project objectpoints with calculated cameramatrix and calculate the distances between localised and reprojected points
mean_error = 0
for i in range(len(object_points)):
    imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    print("error for image ", i + 1, ": ", error)
    mean_error += error
print("mean error: {}".format(mean_error / len(object_points)))

undistorted_live_cam(mtx, dist)
