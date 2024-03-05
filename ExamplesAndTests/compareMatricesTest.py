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

# Example matrices
template_matrix = [
    [0, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 1]
]

example_matrix = [
    [1, 1, 1, 1],
    [1, 1, 1, 0],
    [1, 1, 0, 1]
]

# Compare matrices
score = compare_matrices(template_matrix, example_matrix)
print("Score:", score)
