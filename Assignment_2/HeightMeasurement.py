import numpy as np
import cv2

class HeightMeasurement(object):
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

    def findHorizen(self):
        while True:
            cv2.imshow('image', self.editImage)
            cv2.waitKey(1)
            if len(self.line_coordinates) == 4:
                break
            
    def findVanishingPoint(self):
        while True:
            cv2.imshow('image', self.editImage)
            cv2.waitKey(1)
            if len(self.line_coordinates) == 4:
                break

if __name__ == '__main__':
    draw_line_widget = HeightMeasurement('images/table_bottle_01.jpg')