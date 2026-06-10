import cv2
import numpy as np
import json

# Load the template image
template_img_path = 'Template.jpg'
example_img_path = 'ExampleAnswersCircled2.jpg' 

# Load the example image
template_img = cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE)
example_img = cv2.imread(example_img_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded properly
if template_img is None:
    raise FileNotFoundError("The template image file was not found.")
if example_img is None:
    raise FileNotFoundError("The example image file was not found.")

# Apply a binary threshold to the grayscale image
_, thresh_template = cv2.threshold(template_img, 128, 255, cv2.THRESH_BINARY_INV)
_, thresh_example = cv2.threshold(example_img, 128, 255, cv2.THRESH_BINARY_INV)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# Load the configuration file
config = load_config('config.json')

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

# Initialize the results matrix
num_questions = 6
num_choices = 4
foundCircles = np.zeros((num_questions, num_choices), dtype=int)
results = np.zeros((num_questions, num_choices), dtype=int)

proximity = 30

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

# print(results)
# print()
# print(foundCircles)
# print()

# Remove circled answers
for i in range(num_questions):
    for j in range(num_choices):
        if(foundCircles[i,j] == 1):
            results[i,j] = 0

print(results)