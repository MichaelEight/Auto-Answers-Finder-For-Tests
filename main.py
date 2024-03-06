## IMPORTS
import os
import cv2
import numpy as np
import json
from datetime import datetime

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
        quit()

correct_answers_filename = "PoprawneOdpowiedzi.txt"
correct_answers_matrix = create_matrix_for_correct_answers(correct_answers_filename)

if len(correct_answers_matrix) > 48:
    print("Za dużo odpowiedzi! Mamy tylko 48 miejsc.")
    exit()
print("Odpowiedzi załadowane! Przygotowywanie prac...")

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

print("Prace załadowane! Przygotowywanie do korekty orientacji i skalowania...")

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

print("Prace poprawione! Rozpoczynanie wykrywania odpowiedzi...")

### IMAGE ANSWERS DETECTION
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# Load the configuration file
config = load_config('config.json')

num_choices = 4 # A B C D
proximity = config['image_processing']['circle_proximity_range'] # Constant for circle detection range
marking_threshold = config['image_processing']['marking_threshold']
marking_threshold_factor = config['image_processing']['marking_threshold_factor']
circle_proximity_range = config['image_processing']['circle_proximity_range']
box_width, box_height = config['image_processing']['box_size']
num_questions = len(correct_answers_matrix)

id_box_x, id_box_y = config['index']['starting_box_position']
id_box_width, id_box_height = config['index']['box_size']
id_box_offset_x, id_box_offset_y = config['index']['offset']

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

def is_marked(box, isCheckingId = False):
    global id_box_width, id_box_height, marking_threshold_factor, box_width, box_height

    threshold = marking_threshold_factor

    if(isCheckingId):
        threshold *= id_box_width * id_box_height
    else:
        threshold *= box_width * box_height

    non_white_pixels = np.sum(box < 255)  # Count non-white pixels
    return non_white_pixels < threshold

def compare_matrices(template_matrix, example_matrix):
    score = 0
    for row_template, row_example in zip(template_matrix, example_matrix):
        for elem_template, elem_example in zip(row_template, row_example):
            if elem_example == 1 and elem_template == 0:
                score -= 1  # Subtract 1 point for mismatched positions where Example has "1" and Template has "0"
            elif elem_template == 1 and elem_example == 1:
                score += 1  # Add 1 point if both matrices have "1" at the same position
        score = max(0, score)  # Cap the score at 0 if it becomes negative
    return score

def overlay_rectangle(img, top_left, bottom_right, color, opacity=0.7):
    """
    Overlays a semi-transparent rectangle on the image.
    
    Parameters:
    - img: The image to overlay on.
    - top_left: Tuple (x, y) of the top left corner of the rectangle.
    - bottom_right: Tuple (x, y) of the bottom right corner of the rectangle.
    - color: Tuple (B, G, R) specifying the color of the rectangle.
    - opacity: Opacity of the overlay.
    """
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

# Constants for colors in BGR format
LIGHT_GREEN = (0, 255, 0)
LIGHT_RED = (0, 0, 255)
GRAY = (128, 128, 128)

students_ids_array = []
students_results_array = []
students_scored_points_array = []

# Analyze all images
for index, paper_to_check in enumerate(paper_to_check_image_aligned_array): 
    print(f"Analizowanie pracy {index+1} z {len(paper_to_check_image_aligned_array)}")
    # Apply a binary threshold to the grayscale image
    _, thresh_example = cv2.threshold(paper_to_check, 128, 255, cv2.THRESH_BINARY_INV)
    paper_to_check_color = cv2.cvtColor(paper_to_check, cv2.COLOR_GRAY2BGR)  # Convert to BGR to apply colored overlay

    # DETECT STUDENT'S ID
    student_id = ""
    for col in range(7):
        for row in range(10):
            x = id_box_x + col * id_box_offset_x - id_box_width // 2
            y = id_box_y + row * id_box_offset_y - id_box_height // 2
            box = thresh_example[y:y+id_box_height, x:x+id_box_width]

            if is_marked(box, True):
                row = row + 1
                if row == 10:
                    row = 0
                student_id += str(row)  # Append the detected digit

                # Draw a green rectangle around the detected box
                overlay_rectangle(paper_to_check_color, (x, y), (x + id_box_width, y + id_box_height), LIGHT_GREEN)

                break
        else:
            # If no mark was detected in this column, append a space
            student_id += " "

    student_id = student_id.strip()
    if(" " in student_id or len(student_id) < 6):
        print(f"Błąd w odczycie indeksu dla pracy nr {index+1}")
        student_id = "------"

    students_ids_array.append(student_id)

    # DETECT MARKED (AND CIRCLED) ANSWERS, RETURN ARRAY
    foundCircles = np.zeros((num_questions, num_choices), dtype=int)
    results = np.zeros((num_questions, num_choices), dtype=int) 

    safety_index = 0

    # Detected circles
    for question in config['questions']:
        question_num = question['number'] - 1  # Adjust for zero-indexing
        if question_num >= num_questions: # Skip questions if it exceeds 
            continue  # Skip this question

        for choice in question['choices']:
            choice_num = ord(choice['label']) - ord('A')  # Convert 'A', 'B', 'C', 'D' to 0, 1, 2, 3
            center_x, center_y = choice['center']

            # Calculate the top-left corner of the box
            x_start = center_x - box_width // 2
            y_start = center_y - box_height // 2

            # Extract the box regions from both template and example images
            box_example = thresh_example[y_start:y_start+box_height, x_start:x_start+box_width]

            # Check if the box is circled first
            if is_circled(thresh_example, center_x, center_y, box_width, proximity, 8000):
                foundCircles[question_num, choice_num] = 0
            # If not circled, check if it is marked
            elif is_marked(box_example):
                foundCircles[question_num, choice_num] = 1

        safety_index += 1
        if safety_index >= num_questions * 4:
            print("safety index exceeded 1")
            break
        
    safety_index = 0
    # Detect marked answers
    for question in config['questions']:
        question_num = question['number'] - 1
        for choice in question['choices']:
            choice_num = ord(choice['label']) - ord('A')
            center_x, center_y = choice['center']

            # Calculate the top-left corner of the box
            x_start = center_x - box_width // 2
            y_start = center_y - box_height // 2

            # Extract the box regions from both template and example images
            box_example = thresh_example[y_start:y_start+box_height, x_start:x_start+box_width]

            if is_marked(box_example):
                results[question_num, choice_num] = 1
            
        safety_index += 1
        if safety_index >= num_questions * 4:
            print("safety index exceeded 2")
            break

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

    safety_index = 0

    # SAVE NEW IMAGE AS "CORRECTED_AND_DETECTED" (to later validate detection)
    for question in config['questions']:
        question_num = question['number'] - 1
        for choice in question['choices']:
            choice_num = ord(choice['label']) - ord('A')
            center_x, center_y = choice['center']
            x_start = center_x - box_width // 2
            y_start = center_y - box_height // 2
            x_end = x_start + box_width
            y_end = y_start + box_height
            
            if foundCircles[question_num, choice_num] == 1:
                # Marked and circled
                overlay_rectangle(paper_to_check_color, (x_start, y_start), (x_end, y_end), LIGHT_RED)
            elif results[question_num, choice_num] == 1:
                # Marked but not circled
                overlay_rectangle(paper_to_check_color, (x_start, y_start), (x_end, y_end), LIGHT_GREEN)
            else:
                # Not marked
                overlay_rectangle(paper_to_check_color, (x_start, y_start), (x_end, y_end), GRAY)
    
        safety_index += 1
        if safety_index >= num_questions:
            break

    # Save the modified image
    corrected_image_path = f"PracePrzeanalizowane/corrected_and_detected_{index}.jpg"
    cv2.imwrite(corrected_image_path, paper_to_check_color)

print("Ukończono analizowanie! Wyświetlam odpowiedzi...\n")

## OUTPUT LIST OF POINTS FOR EACH STUDENT'S ID
# TO CONSOLE
for student_id, score in zip(students_ids_array, students_scored_points_array):
        percentage = (score / correct_answers_max_points) * 100
        print(f"{student_id}: {score}, {percentage:.2f}%")

# TO FILE
now = datetime.now() # Current date and time
date_time = now.strftime("%Y-%m-%d %H:%M")

with open("WynikiTestu.txt", "w", encoding='utf-8') as file:
    file.write(f"{date_time}\n")
    file.write(f"Liczba pytań: {num_questions}\n")
    file.write(f"Max punktów: {correct_answers_max_points}\n")
    file.write("Wyniki:\n")
    
    for student_id, score in zip(students_ids_array, students_scored_points_array):
        percentage = (score / correct_answers_max_points) * 100
        file.write(f"{student_id}: {score}, {percentage:.2f}%\n")
