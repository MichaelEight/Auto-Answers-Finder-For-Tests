## IMPORTS
import os

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
print("Image files found:")
for path in paper_to_check_path_array:
    print(path)

## GET PATH FOR TEMPLATE
paper_template_path = 'Template.jpg'

## PERFORM ORIENTATION AND SIZE CORRECTIONS

## FOR EACH IMAGE:
# DETECT MARKED (AND CIRCLED) ANSWERS + STUDENT'S ID, RETURN ARRAY
# COMPARE ARRAY TO LOADED ANSWERS
# CALCULATE POINTS
# SAVE NEW IMAGE AS "CORRECTED_AND_DETECTED" (to later validate detection)

## OUTPUT LIST OF POINTS FOR EACH STUDENT'S ID
# TO CONSOLE
# TO FILE
