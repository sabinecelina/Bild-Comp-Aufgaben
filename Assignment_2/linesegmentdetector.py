import numpy as np
import cv2

img = cv2.imread('images/table_bottle_01.jpg',0)
lineDetector = cv2.createLineSegmentDetector(0)
lines = lineDetector.detect(img)[0]
final_img = lineDetector.drawSegments(img, lines)
cv2.imshow("LSD", final_img)
cv2.waitKey(0)