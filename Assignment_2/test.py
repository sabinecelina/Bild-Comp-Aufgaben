import numpy as np
import cv2
import operator


class DrawLineWidget(object):
    def __init__(self, image, scale=1):
        self.lineColor = (0, 0, 0)
        self.mainImage = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)
        self.editImage = self.mainImage.copy()
        self.image_coordinates = []
        self.line_coordinates = []
        self.drawingLine = False
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.eventHandler)

    def on_mouse_click(self, x, y):
        match self.drawingLine:
            case True:
                self.image_coordinates.append((x, y))
                cv2.line(self.editImage, self.image_coordinates[0], self.image_coordinates[1], self.lineColor, 2)
                self.line_coordinates.append(self.image_coordinates)
                self.mainImage = self.editImage
                self.drawingLine = False
            case False:
                self.image_coordinates = [(x, y)]
                self.drawingLine = True

    def on_mouse_move(self, x, y):
        if self.drawingLine:
            self.editImage = self.mainImage.copy()
            cv2.line(self.editImage, self.image_coordinates[0], (x, y), self.lineColor, 2)

    def eventHandler(self, event, x, y, flags, parameters):
        match event:
            case cv2.EVENT_LBUTTONDOWN:
                self.on_mouse_click(x, y)
            case cv2.EVENT_MOUSEMOVE:
                self.on_mouse_move(x, y)

    def set_line_color(self, color):
        self.lineColor = color

    def get_image(self):
        return self.editImage

    def get_lines(self):
        return self.line_coordinates


def get_intersection_point(pointA1, pointB1, pointA2, pointB2):
    intersection_point = np.cross(np.cross(A1, B1), np.cross(A2, B2))
    intersection_point = intersection_point / intersection_point[2]
    return round(intersection_point[0]), round(intersection_point[1])


def get_image_with_vanishing_points(image, vanishing_points):
    height, width, _ = image.shape
    min_world_x = min(min(vanishing_points)[0], 0)
    max_world_x = max(max(vanishing_points)[0], width)
    min_world_y = min(min(vanishing_points, key=lambda x: x[1])[1], 0)
    max_world_y = max(max(vanishing_points, key=lambda x: x[1])[1], height)
    border = 50
    world_width = max_world_x - min_world_x + (border * 2)
    world_height = max_world_y - min_world_y + (border * 2)
    world_img = np.zeros((world_height, world_width, 3), np.uint8)

    origin = (abs(min_world_x) + border, abs(min_world_y) + border)
    v_x = tuple(map(operator.add, vanishing_points[0], origin))
    v_y = tuple(map(operator.add, vanishing_points[1], origin))

    world_img[origin[1]:origin[1] + height, origin[0]:origin[0] + width, :] = image
    cv2.circle(world_img, v_x, 20, (255, 0, 255), -1)
    cv2.circle(world_img, v_y, 10, (0, 0, 255), -1)
    cv2.line(world_img, v_x, v_y, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(world_img, v_x, 20, (255, 0, 255), -1)
    cv2.circle(world_img, v_y, 10, (0, 0, 255), -1)
    cv2.line(world_img, v_x, v_y, (0, 0, 255), 3, cv2.LINE_AA)

    return world_img, v_x, v_y


def get_distance(point1, point2):
    difference = np.array((point2[0] - point1[0], point2[1] - point1[1]))
    return np.linalg.norm(difference)


if __name__ == '__main__':
    draw_line_widget = DrawLineWidget(cv2.imread('images/table_bottle_01.jpg'), scale=0.4)
    draw_line_widget.set_line_color((255, 50, 0))
    while True:
        cv2.imshow('image', draw_line_widget.get_image())
        cv2.waitKey(1)
        if len(draw_line_widget.get_lines()) == 2:
            draw_line_widget.set_line_color((50, 255, 0))
        if len(draw_line_widget.get_lines()) == 4:
            break
    lines = draw_line_widget.get_lines()

    A1 = np.array([lines[0][0][0], lines[0][0][1], 1])
    B1 = np.array([lines[0][1][0], lines[0][1][1], 1])
    A2 = np.array([lines[1][0][0], lines[1][0][1], 1])
    B2 = np.array([lines[1][1][0], lines[1][1][1], 1])
    point1 = get_intersection_point(A1, B1, A2, B2)
    A1 = np.array([lines[2][0][0], lines[2][0][1], 1])
    B1 = np.array([lines[2][1][0], lines[2][1][1], 1])
    A2 = np.array([lines[3][0][0], lines[3][0][1], 1])
    B2 = np.array([lines[3][1][0], lines[3][1][1], 1])
    point2 = get_intersection_point(A1, B1, A2, B2)

    vanishing_points = [point1, point2]
    world_img, v_x, v_y = get_image_with_vanishing_points(draw_line_widget.get_image(), vanishing_points)

    draw_line_widget = DrawLineWidget(world_img)
    draw_line_widget.set_line_color((50, 50, 255))
    while True:
        cv2.imshow('image', draw_line_widget.get_image())
        cv2.waitKey(1)
        if len(draw_line_widget.get_lines()) == 1:
            draw_line_widget.set_line_color((30, 100, 200))
        if len(draw_line_widget.get_lines()) == 2:
            break
    lines = draw_line_widget.get_lines()

    b = np.array([lines[0][0][0], lines[0][0][1], 1])
    r = np.array([lines[0][1][0], lines[0][1][1], 1])
    b_0 = np.array([lines[1][0][0], lines[1][0][1], 1])
    t_0 = np.array([lines[1][1][0], lines[1][1][1], 1])
    v_x = np.array([v_x[0], v_x[1], 1])
    v_y = np.array([v_y[0], v_y[1], 1])

    v = np.cross(np.cross(b, b_0), np.cross(v_x, v_y))
    v = v / v[2]
    t = np.cross(np.cross(v, t_0), np.cross(r, b))
    t = t / t[2]

    h_ = get_distance(b, t)
    r_ = get_distance(b, r)

    image_cross_ratio = h_ / r_

    print("h: ", h_, "r: ", r_)
    print("image_cross_ratio: ", image_cross_ratio)
    print(26 / image_cross_ratio)

    t = (round(t[0]), round(t[1]))
    cv2.destroyAllWindows()
    image = draw_line_widget.get_image()

    cv2.circle(image, t, 10, (0, 0, 255), -1)
    cv2.imshow('image', image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
