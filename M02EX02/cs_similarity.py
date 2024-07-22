import numpy as np


def cp_cs_similarity(vector1, vector2):
    return (vector1 @ vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))


def main():
    vector1 = np.array([1, 2, 3, 4])
    vector2 = np.array([1, 0, 3, 0])
    print(f"Cosine similarity: {cp_cs_similarity(vector1, vector2)}")


if __name__ == "__main__":
    main()
