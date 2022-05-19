import cv2
import numpy as np


def visualize_sift_keypoints(_img, keypoint_one):
    # Visualize the SIFT keypoints
    keypointimage = cv2.drawKeypoints(_img, keypoint_one, None, color=(0, 255, 0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    height, width, _ = keypointimage.shape
    imS = cv2.resize(keypointimage, (int(height / 8), int(width / 8)))
    return imS


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

imgL = visualize_sift_keypoints(imgL, keypoint_one)
imgM = visualize_sift_keypoints(imgM, keypoint_two)
imgR = visualize_sift_keypoints(imgR, keypoint_three)

numpy_vertical = np.hstack((imgR, imgM, imgL))
cv2.imshow('Numpy Vertical', numpy_vertical)
cv2.waitKey()


def find_matching_points(keypoint_one, keypoint_two):
    # match the keypoints using a FlannBasedMatcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptor_one, descriptor_two, descriptor_three, k=2)
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
    return pts1, pts2
