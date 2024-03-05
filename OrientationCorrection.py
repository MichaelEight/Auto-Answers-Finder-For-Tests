import cv2
import numpy as np

template_path = 'Template.jpg'
example_use_path = 'ExampleAnswersCircled2.jpg'
aligned_img_path = 'Aligned_ExampleUse.jpg'

# Load the images
template_img = cv2.imread(template_path, 0)  # Load in grayscale
example_img = cv2.imread(example_use_path, 0)  # Load in grayscale

# Initialize ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(template_img, None)
keypoints2, descriptors2 = orb.detectAndCompute(example_img, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches (for visual inspection, if needed)
matched_img = cv2.drawMatches(template_img, keypoints1, example_img, keypoints2, matches[:10], None, flags=2)

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Find homography
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

# Use homography
height, width = template_img.shape
aligned_img = cv2.warpPerspective(example_img, h, (width, height))

# Save the transformed image
cv2.imwrite(aligned_img_path, aligned_img)

aligned_img_path
