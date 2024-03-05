## IMPORTS
import os
import cv2
import numpy as np

## LOAD TXT FILE WITH CORRECT ANSWERS
def load_file_with_answers(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        matrix = []
        for line_number, line in enumerate(lines, 1):
            line = line.strip()
            if len(line) != 4:
                raise ValueError(f"Line {line_number} has invalid length. It should be 4 characters long.")
            if not all(char in '01' for char in line):
                raise ValueError(f"Line {line_number} contains invalid characters. Only '0' and '1' are allowed.")
            matrix.append(list(map(int, line)))
        return matrix

def create_matrix_for_correct_answers(filename):
    try:
        data = load_file_with_answers(filename)
        return data
    except Exception as e:
        print("Error:", e)
        return None

correct_answers_filename = "PoprawneOdpowiedzi.txt"
correct_answers_matrix = create_matrix_for_correct_answers(correct_answers_filename)

# CALCULATE MAX POINTS
def calculate_max_points(matrix):
    if not matrix:
        return None
    
    max_points = 0
    for row in matrix:
        for cell in row:
            if cell == 1:
                max_points += 1
    return max_points

correct_answers_max_points = calculate_max_points(correct_answers_matrix)

## GET PATHS FOR ALL IMAGES IN THE FOLDER
def find_image_files():
    image_extensions = ['.jpg', '.jpeg', '.png']  # Add more if needed
    image_files = []
    for root, dirs, files in os.walk("PraceDoSprawdzenia"):
        for file in files:
            _, extension = os.path.splitext(file)
            if extension.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    return image_files

paper_to_check_path_array = find_image_files()

## GET PATH FOR TEMPLATE AND LOAD IT
paper_template_path = 'Template.jpg'
paper_template_image = cv2.imread(paper_template_path, 0)  # Load in grayscale

## LOAD ALL REMAINING IMAGES
def load_images_grayscale(image_paths):
    grayscale_images = []
    for path in image_paths:
        img = cv2.imread(path, 0)  # Load in grayscale
        grayscale_images.append(img)
    return grayscale_images

paper_to_check_image_array = load_images_grayscale(paper_to_check_path_array)

## PERFORM ORIENTATION AND SIZE CORRECTIONS
def find_keypoints_and_descriptors(image):
    orb = cv2.ORB_create() # Initialize ORB detector
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

template_keypoints, template_descriptors = find_keypoints_and_descriptors(paper_template_image)

paper_to_check_keypoints = []
paper_to_check_descriptors = []

for img in paper_to_check_image_array:
    keypoints, descriptors = find_keypoints_and_descriptors(img)
    paper_to_check_keypoints.append(keypoints)
    paper_to_check_descriptors.append(descriptors)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # Create BFMatcher object

all_matches = []

for descriptors in paper_to_check_descriptors:
    matches = bf.match(template_descriptors, descriptors)
    all_matches.append(matches)

# Process matches
for i, matches in enumerate(all_matches):
    # Draw first 10 matches
    matched_img = cv2.drawMatches(paper_template_image, template_keypoints, paper_to_check_image_array[i], paper_to_check_keypoints[i], matches[:10], None, flags=2)
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for j, match in enumerate(matches):
        points1[j, :] = template_keypoints[match.queryIdx].pt
        points2[j, :] = paper_to_check_keypoints[i][match.trainIdx].pt
    
    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography
    height, width = paper_template_image.shape
    aligned_img = cv2.warpPerspective(paper_to_check_image_array[i], h, (width, height))
    
    # Save the transformed image
    aligned_img_path = f"PraceZorientowane/aligned_image_{i}.jpg"
    cv2.imwrite(aligned_img_path, aligned_img)

## FOR EACH IMAGE:
# DETECT MARKED (AND CIRCLED) ANSWERS + STUDENT'S ID, RETURN ARRAY
# COMPARE ARRAY TO LOADED ANSWERS
# CALCULATE POINTS
# SAVE NEW IMAGE AS "CORRECTED_AND_DETECTED" (to later validate detection)

## OUTPUT LIST OF POINTS FOR EACH STUDENT'S ID
# TO CONSOLE
# TO FILE
