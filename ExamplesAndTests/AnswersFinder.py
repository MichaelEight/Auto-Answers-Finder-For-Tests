import cv2
import numpy as np
import json

# Load the template image
template_img_path = 'Template.jpg'
example_img_path = 'ExampleAnswers.jpg' 

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

def is_marked(box):
    # Simple heuristic: if the number of non-white pixels exceeds a threshold, it's marked
    # threshold = config['marking_threshold']
    box_width, box_height = config['box_size']
    threshold = 0.95 * box_width * box_height
    non_white_pixels = np.sum(box < 255)  # Count non-white pixels
    return non_white_pixels > threshold

# Initialize the results matrix
num_questions = 4
num_choices = 4
results = np.ones((num_questions, num_choices), dtype=int)

# Process each box specified in the configuration
for box_info in config['box_centers']:
    question_num = box_info['question'] - 1  # Subtract 1 to convert to 0-index
    choice_num = ord(box_info['choice']) - ord('A')  # Convert 'A', 'B', 'C', 'D' to 0, 1, 2, 3
    center_x, center_y = box_info['center']
    box_width, box_height = config['box_size']
    
    # Calculate the top-left corner of the box
    x_start = center_x - box_width // 2
    y_start = center_y - box_height // 2
    
    # Extract the box regions from both template and example images
    box_template = thresh_template[y_start:y_start+box_height, x_start:x_start+box_width]
    box_example = thresh_example[y_start:y_start+box_height, x_start:x_start+box_width]
    
    # Determine if the box is marked
    if is_marked(box_example):
        results[question_num, choice_num] = 0

print(results)
