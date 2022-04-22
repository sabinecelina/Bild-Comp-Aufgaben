import numpy as np
import cv2
from math import *
import operator

class DrawLineWidget(object):
    def __init__(self, path):
        self.mainImage = cv2.resize(cv2.imread(path), dsize=(0,0), fx=0.4, fy=0.4)
        self.editImage = self.mainImage.copy()
        self.image_coordinates = []
        self.line_coordinates = []
        self.drawingLine = False
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.eventHandler)

    def onMouseClick(self, x, y):
        match self.drawingLine:
            case True:
                self.image_coordinates.append((x,y))
                cv2.line(self.editImage, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
                self.line_coordinates.append(self.image_coordinates)
                self.mainImage = self.editImage
                self.drawingLine = False
            case False:
                self.image_coordinates = [(x,y)]
                self.drawingLine = True

    def onMouseMove(self, x, y):
        if self.drawingLine:
            self.editImage = self.mainImage.copy()
            cv2.line(self.editImage, self.image_coordinates[0], (x, y), (36,255,12), 2)

    def eventHandler(self, event, x, y, flags, parameters):
        match event:
            case cv2.EVENT_LBUTTONDOWN:
                self.onMouseClick(x, y)
            case cv2.EVENT_MOUSEMOVE:
                self.onMouseMove(x, y)

    def getImage(self):
        return self.editImage

    def getLines(self):
        return self.line_coordinates

def getIntersectionPoint(pointA1, pointB1, pointA2, pointB2):
    intersection_point = np.cross(np.cross(A1, B1), np.cross(A2, B2))
    intersection_point = intersection_point / intersection_point[2]
    return (round(intersection_point[0]), round(intersection_point[1]))


if __name__ == '__main__':
    draw_line_widget = DrawLineWidget('images/table_bottle_01.jpg')
    while True:
        cv2.imshow('image', draw_line_widget.getImage())
        cv2.waitKey(1)
        if len(draw_line_widget.getLines()) == 4:
            break
    lines = draw_line_widget.getLines()

    A1 = np.array([lines[0][0][0],lines[0][0][1],1])
    B1 = np.array([lines[0][1][0],lines[0][1][1],1])
    A2 = np.array([lines[1][0][0],lines[1][0][1],1])
    B2 = np.array([lines[1][1][0],lines[1][1][1],1])
    point1 = getIntersectionPoint(A1, B1, A2, B2)
    A1 = np.array([lines[2][0][0],lines[2][0][1],1])
    B1 = np.array([lines[2][1][0],lines[2][1][1],1])
    A2 = np.array([lines[3][0][0],lines[3][0][1],1])
    B2 = np.array([lines[3][1][0],lines[3][1][1],1])
    point2 = getIntersectionPoint(A1, B1, A2, B2)

    vanishing_points = [point1, point2]

    height, width, _ = draw_line_widget.getImage().shape
    min_world_x = min(min(vanishing_points)[0], 0)
    max_world_x = max(max(vanishing_points)[0], width)
    min_world_y = min(min(vanishing_points, key=lambda x: x[1])[1], 0)
    max_world_y = max(max(vanishing_points, key=lambda x: x[1])[1], height)
    border = 50
    world_width = max_world_x - min_world_x + (border*2)
    world_height = max_world_y - min_world_y + (border*2)
    world_img = np.zeros((world_height, world_width, 3), np.uint8)

    origin = (abs(min_world_x) + border, abs(min_world_y) + border)
    vanishing_point1 = tuple(map(operator.add, vanishing_points[0], origin))
    vanishing_point2 = tuple(map(operator.add, vanishing_points[1], origin))
    
    world_img[origin[1]:origin[1]+height, origin[0]:origin[0]+width, :] = draw_line_widget.getImage()
    cv2.circle(world_img, vanishing_point1, 20, (255, 0, 255), -1)
    cv2.circle(world_img, vanishing_point2, 10, (0, 0, 255), -1)
    cv2.line(world_img, vanishing_point1, vanishing_point2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow('image', world_img)
    cv2.waitKey(0)