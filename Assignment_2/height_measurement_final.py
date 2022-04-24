import numpy as np
import cv2
import operator


def drawLine(image, lineColor=(255, 255, 255)):
    def mouseEventHandler(event, x, y, flags, parameters):
        def on_mouse_click(x, y):
            nonlocal editImage, line_coordinates, drawing
            if drawing:
                line_coordinates.append((x, y))
                cv2.line(
                    editImage, line_coordinates[0], line_coordinates[1], lineColor, 2
                )
            else:
                line_coordinates.append((x, y))
                drawing = True

        def on_mouse_move(x, y):
            nonlocal editImage, line_coordinates, drawing
            if drawing:
                editImage = image.copy()
                cv2.line(editImage, line_coordinates[0], (x, y), lineColor, 2)

        match event:
            case cv2.EVENT_LBUTTONDOWN:
                on_mouse_click(x, y)
            case cv2.EVENT_MOUSEMOVE:
                on_mouse_move(x, y)

    editImage = image.copy()
    line_coordinates = []
    drawing = False
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", mouseEventHandler)
    while True:
        cv2.imshow("image", editImage)
        cv2.waitKey(1)
        if len(line_coordinates) == 2:
            break
    line = np.array(
        [
            [line_coordinates[0][0], line_coordinates[0][1]],
            [line_coordinates[1][0], line_coordinates[1][1]],
        ]
    )
    cv2.destroyAllWindows()
    return line, editImage


image = cv2.imread("images/table_bottle_01.jpg")
line, image = drawLine(image)
print(line)


