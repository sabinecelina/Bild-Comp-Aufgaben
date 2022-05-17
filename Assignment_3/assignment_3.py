import cv2

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

# Visualize the SIFT keypoints
keypointimage = cv2.drawKeypoints(imgL, keypoint_one, None, color=(0, 255, 0),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
height, width, _ = keypointimage.shape
imS = cv2.resize(keypointimage, (int(height / 4), int(width / 4)))
cv2.imshow('SIFT', imS)
cv2.waitKey()

# match the keypoints using a FlannBasedMatcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptor_one, descriptor_two, descriptor_three, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]
pts1 = []
pts2 = []
