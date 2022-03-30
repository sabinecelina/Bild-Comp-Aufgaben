# Assignment #01 - Camera Calibration

## Task

Implement and execute the camera calibration process with Python and OpenCV. Based on the found camera parameters implement a script that displays undistorted camera images. Write a readme file that explains what you have done and how to run the script and what the script does.

#### Write a script that allows capturing a set of images for the calibration. (2 points)

The function `capturePictures()` captures a set of images for calibration. <br\>
We captured the webcam in gray and with the `findChessboardCorners()` function we checked if a chessboard got located. If it's true, we save the picture and wait 5000 ms to move the cessboard for a new picture 
```python
i = 0
for x in range(maxImageCount):
    ret, img = cap.read()
    if ret:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("webcam", gray_img)
        found_corners, corners = cv2.findChessboardCorners(gray_img, (chessboardWidth, chessboardHeight), None)
        if found_corners:
            cv2.imwrite("Assignment_1/images/picture_" + str(x) + ".png", gray_img)
            cv2.waitKey(5000)
    cv2.waitKey(1)
```
#### Print out a calibration pattern and capture a set of images of it. Store all needed data in appropriate data structures or files.

This task is stored in the function `findChessboardCorners()`. <br/>
The function `cv2.cornerSubPix()` detects the corners and save the position in pixel to an array. The function `drawCessboardCorners()` draws the corners.

```python
corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
img_points.append(corners)
cv2.drawChessboardCorners(img, (chessboardWidth, chessboardHeight), corners2, ret)

```
#### Write a script that computes the camera parameters. (1 point)

`calibrateCamera()` returns the cameramatrix (intrinsic parameters), the distortion coefficients and the extrinsics for every picture. 

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("camera matrix: \n ", mtx)
print("distortion coefficients: ", dist)

for i in range(len(rvecs)):
    print("parameters for image: ", i, ": \n")
    print("rotation vector: \n", rvecs[i])
    print("translation vector: \n", tvecs[i])
```

#### Write a script that undistorts the live image from a camera. (1 point)

The function `undistorted_live_cam(cameramatrix, distortion_parameters)` starts the webcam
and corrects every frame with the calculated parameters
```python
h, w = frame.shape[:2]
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imshow('video image', dst)
```