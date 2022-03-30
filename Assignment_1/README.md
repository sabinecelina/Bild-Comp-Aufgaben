# Assignment #01 - Camera Calibration

## Task

Implement and execute the camera calibration process with Python and OpenCV. Based on the found camera parameters implement a script that displays undistorted camera images. Write a readme file that explains what you have done and how to run the script and what the script does.

### Write a script that allows capturing a set of images for the calibration. (2 points)

The function `capturePictures()` captures a set of images for calibration. <br\> 
We captured the webcam in gray and with the `findChessboardCorners()` function we checked if a chessboard got located. If it's true, we save the picture and wait 5000 ms to move the cessboard for a new picture 
```python
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

```
### Print out a calibration pattern and capture a set of images of it. Store all needed data in appropriate data structures or files.

This task is stored in the function `findChessboardCorners()`. <br/>
The function `cv2.cornerSubPix()` stores the corners in an array. The function `drawCessboardCorners()` draws the corners.

### Write a script that computes the camera parameters. (1 point)

`calibrateCamera()` returns the cameramatrix (intrinsic parameters), the distortion coefficients and for every picture the extrinsics. 

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("camera matrix: \n ", mtx)
print("distortion coefficients: ", dist)

for i in range(len(rvecs)):
    print("parameters for image: ", i, ": \n")
    print("rotation vector: \n", rvecs[i])
    print("translation vector: \n", tvecs[i])
```