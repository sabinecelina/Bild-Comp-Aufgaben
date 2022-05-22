# Code based on
# https://www.andreasjakl.com/easily-create-depth-maps-with-smartphone-ar-part-1/
# https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/
# https://www.andreasjakl.com/how-to-apply-stereo-matching-to-generate-depth-maps-part-3/

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def detect_keypoints(image1, image2):
    '''Detects and returns keypoints and descriptors between two images.'''
    sift = cv.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    return keypoints1, keypoints2, descriptors1, descriptors2

def visualize_keypoints(image, keypoints, show=True, write=True):
    '''Visualizes Keypoints in a image. Returns the image with keypoints.'''
    image_keypoints = cv.drawKeypoints(
        image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if write:
        cv.imwrite("generated/sift_keypoints.png", image_keypoints)
    if show:
        cv.namedWindow("Keypoints", cv.WINDOW_GUI_NORMAL)
        cv.imshow("Keypoints", image_keypoints)
    return image_keypoints

def filter_keypoints(matches, keypoints1, keypoints2, limit = 0.5):
    # Code based on
    # Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
    # https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    matchesMask = [[0, 0] for i in range(len(matches))]
    good_matches = []
    points1 = []
    points2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < limit * n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append((m, n))
            points2.append(keypoints2[m.trainIdx].pt)
            points1.append(keypoints1[m.queryIdx].pt)
    return points1, points2, tuple(good_matches)

def match_keypoints(keypoints1, keypoints2, descriptors1, descriptors2, filter_limit = 0.5):
    '''Matches keypoints in both images'''
    FLANN_INDEX_KDTREE = 1
    index_parameters = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_parameters = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_parameters, search_parameters)
    all_matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    points1, points2, good_matches = filter_keypoints(all_matches, keypoints1, keypoints2, filter_limit)
    return points1, points2, good_matches, all_matches

def visualize_keypoint_matches(image1, image2, keypoints1, keypoints2, matches, show=True, write=True):
    draw_parameters = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   flags=cv.DrawMatchesFlags_DEFAULT)
    image_matches = cv.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches, None, **draw_parameters)
    if show:
        cv.namedWindow("Keypoint matches", cv.WINDOW_GUI_NORMAL)
        cv.imshow("Keypoint matches", image_matches)
    if write:
        cv.imwrite("generated/keypoint_matches.png", image_matches)
    return image_matches

def calculate_fundamental_matrix(points1, points2):
    points1 = np.int32(points1)
    points2 = np.int32(points2)
    fundamental_matrix, inliers = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC)
    inlier_points1 = points1[inliers.ravel() == 1]
    inlier_points2 = points2[inliers.ravel() == 1]
    return fundamental_matrix, inlier_points1, inlier_points2


def visualize_epilines(image1, image2, points1, points2, fundamental_matrix):
    # based on https://github.com/keckluis/BildCompAufgaben/blob/main/Aufgabe3/DepthMapEstimation.py
    def drawlines(img1src, img2src, lines, pts1src, pts2src):
        r, c = img1src.shape
        img1color = cv.cvtColor(img1src, cv.COLOR_GRAY2BGR)
        img2color = cv.cvtColor(img2src, cv.COLOR_GRAY2BGR)
        np.random.seed(0)
        for r, pt1, pt2 in zip(lines, pts1src, pts2src):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
            img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 1)
            img1color = cv.circle(img1color, tuple(pt1), 5, color, -1)
            img2color = cv.circle(img2color, tuple(pt2), 5, color, -1)
        return img1color, img2color
    lines1 = cv.computeCorrespondEpilines(
        points2.reshape(-1, 1, 2), 2, fundamental_matrix)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(image1, image2, lines1, points1, points2)
    lines2 = cv.computeCorrespondEpilines(
        points1.reshape(-1, 1, 2), 1, fundamental_matrix)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(image2, image1, lines2, points2, points1)
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.suptitle("Epilines in both images")
    plt.savefig("generated/epilines.png")
    plt.show()

def rectifyImages(image1, image2, points1, points2, fundamental_matrix):
    height1, width1 = image1.shape
    height2, width2 = image2.shape
    _, H1, H2 = cv.stereoRectifyUncalibrated(
        np.float32(points1), np.float32(points2), fundamental_matrix, imgSize=(width1, height1)
    )
    image1_rectified = cv.warpPerspective(image1, H1, (width1, height1))
    image2_rectified = cv.warpPerspective(image2, H2, (width2, height2))
    return image1_rectified, image2_rectified

def visualize_rectified_images(image1_rectified, image2_rectified, show=True, write=True):
    if write:
        cv.imwrite("generated/rectified_1.png", image1_rectified)
        cv.imwrite("generated/rectified_2.png", image2_rectified)
    if show:
        cv.namedWindow("rectified_images", cv.WINDOW_GUI_NORMAL)
        cv.imshow("rectified_images", np.concatenate((image1_rectified, image2_rectified), axis=1))

def calculate_disparity_map(image1_rectified, image2_rectified):
    # Adapted from: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
    # and: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

    # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
    block_size = 11
    min_disp = -128
    max_disp = 128
    # Maximum disparity minus minimum disparity. The value is always greater than zero.
    # In the current implementation, this parameter must be divisible by 16.
    num_disp = max_disp - min_disp
    # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
    # Normally, a value within the 5-15 range is good enough
    uniquenessRatio = 5
    # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
    # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    speckleWindowSize = 200
    # Maximum disparity variation within each connected component.
    # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
    # Normally, 1 or 2 is good enough.
    speckleRange = 2
    disp12MaxDiff = 0

    stereo = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )
    image_disparity_SGBM = stereo.compute(image1_rectified, image2_rectified)

    # Normalize the values to a range from 0..255 for a grayscale image
    image_disparity_SGBM = cv.normalize(image_disparity_SGBM, image_disparity_SGBM, alpha=255,
                                beta=0, norm_type=cv.NORM_MINMAX)
    image_disparity_SGBM = np.uint8(image_disparity_SGBM)
    cv.imwrite("generated/disparity_SGBM_normalized.png", image_disparity_SGBM)
    return image_disparity_SGBM

if __name__ == "__main__":
    image1 = cv.imread('images/left.jpg', cv.IMREAD_GRAYSCALE)
    image2 = cv.imread('images/middle.jpg', cv.IMREAD_GRAYSCALE)

    keypoints1, keypoints2, descriptors1, descriptors2 = detect_keypoints(image1, image2)
    visualize_keypoints(image1, keypoints1, show=True, write=True)
    visualize_keypoints(image2, keypoints2, show=True, write=True)
    points1, points2, good_matches, all_matches = match_keypoints(keypoints1, keypoints2, descriptors1, descriptors2, filter_limit = 0.6)
    visualize_keypoint_matches(image1, image2, keypoints1, keypoints2, good_matches, show=True, write=True)
    fundamental_matrix, inlier_points1, inlier_points2 = calculate_fundamental_matrix(points1, points2)
    visualize_epilines(image1, image2, inlier_points1, inlier_points2, fundamental_matrix)
    image1_rectified, image2_rectified = rectifyImages(image1, image2, points1, points2, fundamental_matrix)
    visualize_rectified_images(image1_rectified, image2_rectified)
    disparity_SGBM = calculate_disparity_map(image1_rectified, image2_rectified)
    cv.namedWindow("disparity map", cv.WINDOW_GUI_NORMAL)
    cv.imshow("disparity map", disparity_SGBM)
    cv.waitKey(0)
    cv.destroyAllWindows()


