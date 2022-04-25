# Assignment #02 - Height measurement

## Task 

Implement a script with Python and OpenCV that calculates the height of an object given a reference object placed on the same plane. Given three exemplary images (images\table_bottle_*.jpg) of a table showing two objects - a bottle and a mug. The bottle has a height* of 26 cm. Calculate the height of the mug in cm. Write a readme file that explains what you have done and how to run the script and what the script does.

* measured by hand as no official source found for the bottle dimensions.

Note: The program starts with the main-method in which all required functions are called.

#### Write a script that allows manually selecting the needed image points for the calculation. (2 points)

At first the user has to choose the lines for calculating the vanishing points manually. 
With the function ```input_line()``` the user can select lines in the image.
This function returns the lines which are needed for calculating the vanishing points.
```python
    lineA1, image = input_line(image, lineColor=(0, 255, 0), thickness=3)
    lineA2, image = input_line(image, lineColor=(0, 255, 0), thickness=3)
    print("""\nIn the second step, mark two different parallel lines on the plane. 
    Those lines should not be parallel to the last two lines you marked.\n""")
    lineB1, image = input_line(image, lineColor=(255, 0, 0), thickness=3)
    lineB2, image = input_line(image, lineColor=(255, 0, 0), thickness=3)
```
Next step is to mark the unknown object from botton to top so the height can be calculated and also mark the reference object where the height is known.
```python
    print("""\nNow mark the object from bottom to top.
    Note that the selection can be difficult depending on the shape of the object due to perspective effects.\n""")
    lineC, image = input_line(image, lineColor=(0, 0, 255), thickness=3)
    print("""\nNow mark the reference object from bottom to top. Note that the selection can be difficult 
    depending on the shape of the object due to perspective effects.\n""")
    lineD, image = input_line(image, lineColor=(0, 150, 200), thickness=3)
```

#### Write a script that computes the vanishing line for the table plane and visualizes it. (2 points)

The function ```get_intersection_point()``` calculates the intersection between to lines.

```insert_point_in_image()``` displays the vanishing points. If the point is not in the image it fills space with black.

#### Write a script that computes the height of the mug. (2 points)

At least the function ```calculate_cross_ratio()``` calculates the cross ratio of the unknown object with the given formula
```python
    v_x = get_intersection_point(
        parallelLinePair_A[0][0], parallelLinePair_A[0][1], parallelLinePair_A[1][0], parallelLinePair_A[1][1])
    v_y = get_intersection_point(
        parallelLinePair_B[0][0], parallelLinePair_B[0][1], parallelLinePair_B[1][0], parallelLinePair_B[1][1])
    b = referenceObject[0]
    r = referenceObject[1]
    b_0 = object[0]
    t_0 = object[1]
    v = get_intersection_point(b, b_0, v_x, v_y)
    t = get_intersection_point(v, t_0, r, b)

    v_z = get_intersection_point(t_0, b_0, r, b)
    crossratio = (np.linalg.norm(t - b) * np.linalg.norm(v_z - r)) / (np.linalg.norm(r - b) *np.linalg.norm(v_z - t))
```

With the reference object the program calculates the height of the unknow object in cm.
```python
    print("\nEnter the height of the reference object in cm as a floating point number.\n")
    input_float = float(input())
    height = cross_ratio * input_float
    print("\n The height of the object is: ", height, " cm")
```
According to our calculations the height should be 9.57cm