## LOAD ALL IMAGES FROM THE FOLDER

## LOAD TXT FILE WITH CORRECT ANSWERS
# CALCULATE MAX POINTS
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

filename = "PoprawneOdpowiedzi.txt"
matrix = create_matrix_for_correct_answers(filename)
print(matrix)


## PERFORM ORIENTATION AND SIZE CORRECTIONS

## FOR EACH IMAGE:
# DETECT MARKED (AND CIRCLED) ANSWERS + STUDENT'S ID, RETURN ARRAY
# COMPARE ARRAY TO LOADED ANSWERS
# CALCULATE POINTS
# SAVE NEW IMAGE AS "CORRECTED_AND_DETECTED" (to later validate detection)

## OUTPUT LIST OF POINTS FOR EACH STUDENT'S ID
# TO CONSOLE
# TO FILE
