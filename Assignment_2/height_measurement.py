import numpy as np
import cv2
from math import *
from DrawLineWidget import DrawLineWidget

if __name__ == '__main__':
    draw_line_widget = DrawLineWidget('images/table_bottle_01.jpg')
    while True:
        cv2.imshow('image', draw_line_widget.getImage())
        cv2.waitKey(1)
        if len(draw_line_widget.getLines()) == 5:
            break
    lines = draw_line_widget.getLines()

    print(lines)
    A1 = np.array([lines[0][0][0],lines[0][0][1],1])
    B1 = np.array([lines[0][1][0],lines[0][1][1],1])
    A2 = np.array([lines[1][0][0],lines[1][0][1],1])
    B2 = np.array([lines[1][1][0],lines[1][1][1],1])
    point = np.cross(np.cross(A1, B1), np.cross(A2, B2))
    point = point / point[2]