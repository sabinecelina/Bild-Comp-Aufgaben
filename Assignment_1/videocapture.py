import numpy as np
import cv2
from math import *

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = int(cap.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT))

title = 'Video image'
cv2.namedWindow(title,  cv2.WINDOW_FREERATIO)

loop, frame = cap.read()
while loop:
    img = np.zeros(frame.shape, np.uint8)
    smaller_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    img[:floor(height/2), :floor(width/2)] = smaller_frame
    img[floor(height/2):, :floor(width/2)] = cv2.flip(smaller_frame, 0)
    img[floor(height/2):, floor(width/2):] = cv2.flip(smaller_frame, -1)
    img[:floor(height/2), floor(width/2):] = cv2.flip(smaller_frame, 1)
    cv2.imshow(title, img)

    loop, frame = cap.read()
    if cv2.waitKey(1) == ord('q'):
        loop = 0

cap.release()
cv2.destroyAllWindows()