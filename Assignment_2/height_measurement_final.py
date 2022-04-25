import numpy as np
import cv2


# lets a user draw a line on a given image
# input: image
# returns: new image, linepoint coordinates as shape (2,2) np.array


def input_line(image, line_color=(255, 255, 255), thickness=1):
    def mouseEventHandler(event, x, y, flags, parameters):
        def on_mouse_click(x, y):
            nonlocal edit_image, line_coordinates, drawing
            if drawing:
                line_coordinates.append((x, y))
                cv2.line(
                    edit_image, line_coordinates[0], line_coordinates[1], line_color, thickness)
            else:
                line_coordinates.append((x, y))
                drawing = True

        def on_mouse_move(x, y):
            nonlocal edit_image, line_coordinates, drawing
            if drawing:
                edit_image = image.copy()
                cv2.line(
                    edit_image, line_coordinates[0], (x, y), line_color, thickness)

        match event:
            case cv2.EVENT_LBUTTONDOWN:
                on_mouse_click(x, y)
            case cv2.EVENT_MOUSEMOVE:
                on_mouse_move(x, y)

    edit_image = image.copy()
    line_coordinates = []
    drawing = False
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", mouseEventHandler)
    while True:
        cv2.imshow("image", edit_image)
        cv2.waitKey(1)
        if len(line_coordinates) == 2:
            break
    line = np.array(
        [[line_coordinates[0][0], line_coordinates[0][1], 1],
         [line_coordinates[1][0], line_coordinates[1][1], 1]])
    return line, edit_image


# inserts Point. If the point is not in the image it fills space with black.
# With the x and y offset the new image coordinates can be calculated.
# input: image, point in image coordinates as (2) np.array
# returns: new image, point in new image coordinates, x and y offset from original image


def insert_point_in_image(image, point, radius=10, color=(255, 255, 255), border=0):
    height, width, _ = image.shape
    x, y = round(point[0]), round(point[1])
    border = radius + border
    # finding min and max x and y values needed for new image
    min_x = min(x - border, 0)
    max_x = max(x + border, width)
    min_y = min(y - border, 0)
    max_y = max(y + border, height)
    # calculate size of new image
    new_image_width = max_x - min_x
    new_image_height = max_y - min_y
    # fill new image with black, draw old image and draw point
    new_image = np.zeros((new_image_height, new_image_width, 3), np.uint8)
    offset = np.array([abs(min_x), abs(min_y), 0])
    new_image[offset[1]:offset[1] + height,
              offset[0]:offset[0] + width, :] = image
    new_point = point + np.array([offset[0], offset[1], 1])
    cv2.circle(new_image, (round(new_point[0]), round(
        new_point[1])), radius, color, -1)
    # return new image, point in new coordinates and the offset from the original image
    return new_image, new_point, offset


def get_intersection_point(a1, a2, b1, b2):
    point = np.cross(np.cross(a1, a2), np.cross(b1, b2))
    point = point / point[2]
    return np.array([point[0], point[1], 1])


def get_new_image(image, v_x, v_y, b, r, b_0, t_0, v, t, v_z):
    image, _, o = insert_point_in_image(
        image, v_x, radius=20, color=(255, 100, 0), border=10)
    v_x, v_y, b, r, b_0, t_0, v, t, v_z = v_x + o, v_y + \
        o, b + o, r + o, b_0 + 0, t_0 + o, v + o, t + o, v_z + o
    offset = o
    image, _, o = insert_point_in_image(
        image, v_y, radius=20, color=(255, 100, 0), border=10)
    v_x, v_y, b, r, b_0, t_0, v, t, v_z = v_x + o, v_y + \
        o, b + o, r + o, b_0 + 0, t_0 + o, v + o, t + o, v_z + o
    offset += o
    image, _, o = insert_point_in_image(
        image, v, radius=20, color=(200, 255, 0), border=10)
    v_x, v_y, b, r, b_0, t_0, v, t, v_z = v_x + o, v_y + \
        o, b + o, r + o, b_0 + 0, t_0 + o, v + o, t + o, v_z + o
    offset += o
    image, _, o = insert_point_in_image(
        image, t, radius=10, color=(100, 200, 255), border=10)
    v_x, v_y, b, r, b_0, t_0, v, t, v_z = v_x + o, v_y + \
        o, b + o, r + o, b_0 + 0, t_0 + o, v + o, t + o, v_z + o
    offset += o
    v_x = (round(v_x[0]), round(v_x[1]))
    v_y = (round(v_y[0]), round(v_y[1]))
    b = (round(b[0]), round(b[1]))
    r = (round(r[0]), round(r[1]))
    b_0 = (round(b_0[0]), round(b_0[1]))
    t_0 = (round(t_0[0]), round(t_0[1]))
    v = (round(v[0]), round(v[1]))
    t = (round(t[0]), round(t[1]))
    v_z = (round(v_z[0]), round(v_z[1]))

    cv2.line(image, (v_x[0], v_x[1]), (v_y[0], v_y[1]), (255, 100, 200), 3)
    cv2.line(image, (v_x[0], v_x[1]), (v[0], v[1]), (255, 100, 200), 3)
    cv2.line(image, (v[0], v[1]), (v_y[0], v_y[1]), (255, 100, 200), 3)
    cv2.line(image, (v[0], v[1]), (b[0], b[1]), (0, 100, 255), 2)
    cv2.line(image, (v[0], v[1]), (t[0], t[1]), (0, 100, 255), 2)
    return image


def calculate_cross_ratio(parallelLine_pair_A, parallelLine_pair_B, reference_object, object, image):
    edit_image = image.copy()
    v_x = get_intersection_point(
        parallelLine_pair_A[0][0], parallelLine_pair_A[0][1], parallelLine_pair_A[1][0], parallelLine_pair_A[1][1])
    v_y = get_intersection_point(
        parallelLine_pair_B[0][0], parallelLine_pair_B[0][1], parallelLine_pair_B[1][0], parallelLine_pair_B[1][1])
    b = reference_object[0]
    r = reference_object[1]
    b_0 = object[0]
    t_0 = object[1]
    v = get_intersection_point(b, b_0, v_x, v_y)
    t = get_intersection_point(v, t_0, r, b)
    v_z = get_intersection_point(t_0, b_0, r, b)
    cross_ratio = (np.linalg.norm(t - b) * np.linalg.norm(v_z - r)) / \
        (np.linalg.norm(r - b) * np.linalg.norm(v_z - t))
    return cross_ratio, get_new_image(edit_image, v_x, v_y, b, r, b_0, t_0, v, t, v_z)


if __name__ == '__main__':
    print("""This program will allow you to determine the height of an object in an image. 
    For this, the image must contain a reference object with a known height. 
    Both objects should ideally be placed on the same plane.\n""")
    print("Do you want to enter a custom image filename? (enter y or n)")
    input_string = str(input())
    path = "images/"
    if input_string == "y" or input_string == "yes":
        print("Enter the image file name with the file extension (.jpg)")
        input_string = str(input())
        path += input_string
    else:
        path += "table_bottle_01.jpg"
    print("""\nIn the first step, mark two parallel lines on the plane. 
    Click to set the start or end point of a line.\n""")
    image = cv2.imread(path)
    original_image = image.copy()
    lineA1, image = input_line(image, line_color=(0, 255, 0), thickness=3)
    lineA2, image = input_line(image, line_color=(0, 255, 0), thickness=3)
    print("""\nIn the second step, mark two different parallel lines on the plane. 
    Those lines should not be parallel to the last two lines you marked.\n""")
    lineB1, image = input_line(image, line_color=(255, 0, 0), thickness=3)
    lineB2, image = input_line(image, line_color=(255, 0, 0), thickness=3)
    print("""\nNow mark the object from bottom to top.
    Note that the selection can be difficult depending on the shape of the object due to perspective effects.\n""")
    lineC, image = input_line(image, line_color=(0, 0, 255), thickness=3)
    print("""\nNow mark the reference object from bottom to top. Note that the selection can be difficult 
    depending on the shape of the object due to perspective effects.\n""")
    lineD, image = input_line(image, line_color=(0, 150, 200), thickness=3)
    cv2.destroyAllWindows()
    linePair_A = np.array([lineA1, lineA2])
    linePair_B = np.array([lineB1, lineB2])
    cross_ratio, image = calculate_cross_ratio(
        linePair_A, linePair_B, lineD, lineC, image)
    print("\nEnter the height of the reference object in cm as a floating point number.\n")
    input_float = float(input())
    height = cross_ratio * input_float
    print("\n The height of the object is: ", height, " cm")

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

