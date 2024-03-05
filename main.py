## IMPORTS
import os
import cv2
import numpy as np
import json

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

### IMAGE ORIENTATION AND SIZE CORRECTION

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

paper_to_check_image_aligned_array = []

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
    
    paper_to_check_image_aligned_array.append(aligned_img)

    # Save the transformed image
    aligned_img_path = f"PraceZorientowane/aligned_image_{i}.jpg"
    cv2.imwrite(aligned_img_path, aligned_img)

### IMAGE ANSWERS DETECTION
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# Load the configuration file
config = load_config('config.json')

num_choices = 4 # A B C D
proximity = 30 # Just some constant
num_questions = len(correct_answers_matrix)

# Apply a binary threshold to the grayscale image
_, thresh_template = cv2.threshold(paper_template_image, 128, 255, cv2.THRESH_BINARY_INV)

def is_circled(img, center_x, center_y, box_size, proximity, dark_pixel_threshold):
    # Define how many pixels larger the exclusion zone should be
    exclusion_border = 7  # Increase this value as needed

    # Calculate the top-left and bottom-right points for the larger square (box + proximity)
    x_start = max(center_x - (box_size // 2 + proximity), 0)
    y_start = max(center_y - (box_size // 2 + proximity), 0)
    x_end = min(center_x + (box_size // 2 + proximity), img.shape[1])
    y_end = min(center_y + (box_size // 2 + proximity), img.shape[0])

    # Extract the larger square region
    region = img[y_start:y_end, x_start:x_end]

    # Create a mask for the actual answer box to exclude it
    mask = np.zeros(region.shape, dtype=np.uint8)
    # Calculate the increased exclusion zone
    exclusion_x_start = proximity - exclusion_border
    exclusion_y_start = proximity - exclusion_border
    exclusion_x_end = exclusion_x_start + box_size + (exclusion_border * 2)
    exclusion_y_end = exclusion_y_start + box_size + (exclusion_border * 2)
    cv2.rectangle(mask, (exclusion_x_start, exclusion_y_start), (exclusion_x_end, exclusion_y_end), 255, -1)

    # Calculate the number of dark pixels outside the answer box area
    dark_pixels_outside_box = np.sum(region[mask == 0] < 128)

    # Define a threshold for considering a box as circled
    # If there are dark pixels around the box exceeding this threshold, it's circled
    #dark_pixel_threshold = 8000  # Adjust this threshold as needed

    return dark_pixels_outside_box > dark_pixel_threshold

def is_marked(box):
    # Simple heuristic: if the number of non-white pixels exceeds a threshold, it's marked
    # threshold = config['marking_threshold']
    box_width, box_height = config['box_size']
    threshold = 0.95 * box_width * box_height
    non_white_pixels = np.sum(box < 255)  # Count non-white pixels
    return non_white_pixels < threshold

def compare_matrices(template_matrix, example_matrix):
    score = 0
    for row_template, row_example in zip(template_matrix, example_matrix):
        for elem_template, elem_example in zip(row_template, row_example):
            if elem_template == 1 and elem_example == 1:
                score += 1  # Add 1 point if both matrices have 1 at the same position
            elif elem_template != elem_example:
                score -= 1  # Subtract 1 point for mismatched positions
        score = max(0, score)  # Cap the score at 0 if it becomes negative
    return score

students_results_array = []
students_scored_points_array = []

# Analyze all images
for paper_to_check in paper_to_check_image_aligned_array:
    # Apply a binary threshold to the grayscale image
    _, thresh_example = cv2.threshold(paper_to_check, 128, 255, cv2.THRESH_BINARY_INV)

    # DETECT MARKED (AND CIRCLED) ANSWERS + STUDENT'S ID, RETURN ARRAY
    foundCircles = np.zeros((num_questions, num_choices), dtype=int)
    results = np.zeros((num_questions, num_choices), dtype=int) 

    # Detected circles
    for box_info in config['box_centers']:
        question_num = box_info['question'] - 1  # Subtract 1 to convert to 0-index
        choice_num = ord(box_info['choice']) - ord('A')  # Convert 'A', 'B', 'C', 'D' to 0, 1, 2, 3
        center_x, center_y = box_info['center']
        box_width, box_height = config['box_size']

        # Calculate the top-left corner of the box
        x_start = center_x - box_width // 2
        y_start = center_y - box_height // 2

        # Extract the box regions from both template and example images
        box_example = thresh_example[y_start:y_start+box_height, x_start:x_start+box_width]

        print(f"Question Number: {question_num}, Choice Number: {choice_num}")

        # Check if the box is circled first
        if is_circled(thresh_example, center_x, center_y, box_width, proximity, 8000):
            foundCircles[question_num, choice_num] = 0
        # If not circled, check if it is marked
        elif is_marked(box_example):
            foundCircles[question_num, choice_num] = 1
        
    # Detect marked answers
    for box_info in config['box_centers']:
        question_num = box_info['question'] - 1  # Subtract 1 to convert to 0-index
        choice_num = ord(box_info['choice']) - ord('A')  # Convert 'A', 'B', 'C', 'D' to 0, 1, 2, 3
        center_x, center_y = box_info['center']
        box_width, box_height = config['box_size']

        # Calculate the top-left corner of the box
        x_start = center_x - box_width // 2
        y_start = center_y - box_height // 2

        # Extract the box regions from both template and example images
        box_example = thresh_example[y_start:y_start+box_height, x_start:x_start+box_width]

        if is_marked(box_example):
            results[question_num, choice_num] = 1

    # Remove circled answers
    for i in range(num_questions):
        for j in range(num_choices):
            if(foundCircles[i,j] == 1):
                results[i,j] = 0

    # Save student's answers as matrix
    students_results_array.append(results)

    # CALCULATE POINTS
    score = compare_matrices(correct_answers_matrix, results)
    students_scored_points_array.append(score)

    # SAVE NEW IMAGE AS "CORRECTED_AND_DETECTED" (to later validate detection)

## OUTPUT LIST OF POINTS FOR EACH STUDENT'S ID
# TO CONSOLE
for score in students_scored_points_array:
    print(score)
# TO FILE
