import numpy as np
import cv2
import operator

#lets a user draw a line on a given image
#input: image
#returns: new image, linepoint coordinates as shape (2,2) np.array
def drawLine(image, lineColor = (255, 255, 255), thickness = 1):
    def mouseEventHandler(event, x, y, flags, parameters):
        def on_mouse_click(x, y):
            nonlocal editImage, line_coordinates, drawing
            if drawing:
                line_coordinates.append((x, y))
                cv2.line(
                    editImage, line_coordinates[0], line_coordinates[1], lineColor, thickness
                )
            else:
                line_coordinates.append((x, y))
                drawing = True

        def on_mouse_move(x, y):
            nonlocal editImage, line_coordinates, drawing
            if drawing:
                editImage = image.copy()
                cv2.line(editImage, line_coordinates[0], (x, y), lineColor, thickness)

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

#inserts Point. If the point is not in the image it fills space with black. 
#With the x and y offset the new image coordinates can be calculated.
#input: image, point in image coordinates as (2) np.array
#returns: new image, point in new image coordinates, x and y offset from original image
def insertPointInImage(image, point, radius = 10, color = (255, 255, 255), border = 0):
    height, width, _ = image.shape
    x, y = point[0], point[1]
    border = radius + border
    # finding min and max x and y values needed for new image
    min_x = min(x - border, 0)
    max_x = max(x + border, width)
    min_y = min(y - border, 0)
    max_y = max(y + border, height)
    # calculate size of new image
    new_image_width = max_x - min_x
    new_image_height = max_y - min_y
    # fill new image with old image and draw point
    new_image = np.zeros((new_image_height, new_image_width, 3), np.uint8)
    offset = np.array([abs(min_x), abs(min_y)])
    new_image[offset[1]:offset[1] + height, offset[0]:offset[0] + width, :] = image
    new_point = point + offset
    cv2.circle(new_image, new_point, radius, color, -1)
    
    return new_image, new_point, offset
    


image = cv2.imread("images/table_bottle_01.jpg")
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
#line, image = drawLine(image)
#cv2.imshow("image", image)
#cv2.waitKey(0)
#print(line)

image, point, offset = insertPointInImage(image, np.array([-50,-50]),radius = 20, color = (255, 50, 50), border = 10)
cv2.imshow("image", image)
print(point, offset)
cv2.waitKey(0)


