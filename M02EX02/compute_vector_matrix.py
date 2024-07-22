import numpy as np


def compute_vector_length(vector):
    len_of_vector = np.sqrt((vector**2).sum())
    return len_of_vector


def compute_dot_product(array1, array2):
    result = array1 @ array2
    return result


def inverse_matrix(matrix):
    result = np.linalg.inv(matrix)
    return result


def main():
    vector1 = np.array([4, 3])
    vector2 = np.array([3, 4])
    matrix1 = np.array([[1, 2], [2, 2]])
    matrix2 = np.array([[1, 1], [2, 2]])

    print(f"Length of vector: {compute_vector_length(vector1)}")
    print(f"Vector multi vector:\n{compute_dot_product(vector1, vector2)}")
    print(f"Matrix multi vector:\n{compute_dot_product(matrix1, vector1)}")
    print(f"Matrix multi matrix:\n{compute_dot_product(matrix1, matrix2)}")
    print(f"Inverse matrix:\n{inverse_matrix(matrix1)}")


if __name__ == "__main__":
    main()
