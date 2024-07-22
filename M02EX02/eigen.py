import numpy as np


def compute_eigen(matrix):
    return np.linalg.eig(matrix)


def main():
    matrix = np.array([[0.9, 0.2], [0.1, 0.8]])

    eigenvalues, eigenvectors = compute_eigen(matrix)
    norm_eigenvectors = eigenvectors / \
        np.linalg.norm(eigenvectors, axis=1).reshape(-1, 1)

    print(f"Eigenvalues:\n{eigenvalues}")
    print(f"Normalize eigenvectors:\n{norm_eigenvectors}")


if __name__ == "__main__":
    main()
