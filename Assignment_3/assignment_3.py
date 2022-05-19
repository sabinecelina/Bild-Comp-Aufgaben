import cv2
import numpy as np


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def draw_epiline(img1, img2, pts1, pts2, F):
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    height, width, _ = img3.shape
    imS = cv2.resize(img3, (int(height / 6), int(width / 6)))
    height, width, _ = img5.shape
    imSV = cv2.resize(img5, (int(height / 6), int(width / 6)))
    cv2.imshow("title", np.concatenate((imS, imSV), axis=1))
    cv2.waitKey(0)


def visualize_sift_keypoints(_img, keypoint_one):
    # Visualize the SIFT keypoints
    keypointimage = cv2.drawKeypoints(_img, keypoint_one, None, color=(0, 255, 0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    height, width, _ = keypointimage.shape
    imS = cv2.resize(keypointimage, (int(height / 8), int(width / 8)))
    return imS


def find_matching_points(keypoint_one, keypoint_two):
    # match the keypoints using a FlannBasedMatcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptor_one, descriptor_two, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            pts2.append(keypoint_two[m.trainIdx].pt)
            pts1.append(keypoint_one[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    print('We found %d matching keypoints in both images.' % len(pts1))

    # Compute the Fundamental Matrix.

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    return pts1, pts2, F


def rectify_images(img1, img2, pts1, pts2, F):
    height_one, width_one = img1.shape
    height_two, width_two = img2.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(width_one, height_one))
    img1_rectified = cv2.warpPerspective(img1, H1, (width_one, height_one))
    img2_rectified = cv2.warpPerspective(img2, H2, (width_two, height_two))
    img1_rectified_resized = cv2.resize(img1_rectified, (int(height_one / 8), int(width_two / 8)))
    img2_rectified_resized = cv2.resize(img2_rectified, (int(height_two / 8), int(width_two / 8)))
    numpy_vertical = np.hstack((img1_rectified_resized, img2_rectified_resized))
    # cv2.imshow("rectified images: ", numpy_vertical)
    # cv2.waitKey(0)
    return img1_rectified, img2_rectified


def createDepthMap(img1_rectified, img2_rectified):
    # Creating an object of StereoSGBM algorithm
    stereo = cv2.StereoSGBM_create(minDisparity=0,
                                   numDisparities=64,
                                   blockSize=8,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=10,
                                   speckleRange=8)

    # Calculate disparity using the chosen stereo algorithm
    disp = stereo.compute(img1_rectified, img2_rectified).astype(np.float32)

    # Normalize the disparity map in order to display it
    disp = cv2.normalize(disp, 0, 255, cv2.NORM_MINMAX)
    depth_map = np.uint8(disp)
    disp_resized = cv2.resize(disp, (int(disp.shape[0] / 8), int(disp.shape[1] / 8)))
    # Display the disparity map
    cv2.imshow("disparity", disp_resized)
    cv2.waitKey(0)
    return depth_map


def create_one_depth_map(depth_maps):
    depth_map = np.zeros(depth_maps[0].shape)
    w, h = depth_maps[0].shape
    for x in range(w - 1):
        for y in range(h - 1):
            depth_map[x, y] = (int(depth_maps[0][x, y]) + int(depth_maps[1][x, y]) + int(depth_maps[2][x, y])) / 255 / 3


# load images
imgL = cv2.imread("images/left.jpg", cv2.IMREAD_GRAYSCALE)
imgM = cv2.imread("images/middle.jpg", cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread("images/right.jpg", cv2.IMREAD_GRAYSCALE)

# find the keypoints and descriptors using SIFT_create
sift_object = cv2.SIFT_create()
keypoint_one, descriptor_one = sift_object.detectAndCompute(imgL, None)
keypoint_two, descriptor_two = sift_object.detectAndCompute(imgM, None)
keypoint_three, descriptor_three = sift_object.detectAndCompute(imgR, None)
print('We found %d keypoints in the left image.' % len(keypoint_one))
print('We found %d keypoints in the middle image.' % len(keypoint_two))
print('We found %d keypoints in the right image.' % len(keypoint_three))

# visualize sift keypoints
imgLeft = visualize_sift_keypoints(imgL, keypoint_one)
imgMiddle = visualize_sift_keypoints(imgM, keypoint_two)
imgRight = visualize_sift_keypoints(imgR, keypoint_three)

numpy_vertical = np.hstack((imgLeft, imgMiddle, imgRight))
# cv2.imshow('Numpy Vertical', numpy_vertical)
# cv2.waitKey()

# find matching points between imgL and imgM
pts_1, pts_2, F1 = find_matching_points(keypoint_one, keypoint_two)
rectified_images = rectify_images(imgL, imgM, pts_1, pts_2, F1)
depth_map_one = createDepthMap(rectified_images[0], rectified_images[1])
# draw_epiline(rectify_images[0], rectify_images[1], pts_1, pts_2, F1)
# find matching points between imgL and imgR
pts_3, pts_4, F2 = find_matching_points(keypoint_one, keypoint_three)
rectified_images = rectify_images(imgL, imgR, pts_3, pts_4, F2)
depth_map_two = createDepthMap(rectified_images[0], rectified_images[1])
# find matching points between imgM and imgR
pts_5, pts_6, F3 = find_matching_points(keypoint_two, keypoint_three)
rectified_images = rectify_images(imgM, imgR, pts_5, pts_6, F3)
depth_map_three = createDepthMap(rectified_images[0], rectified_images[1])
