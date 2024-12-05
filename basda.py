import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# create matrix F
F = np.array([[1, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0,], [1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1]] )

# use np.linalg.svd, to find the svd of F
U, Sigma, VT = np.linalg.svd(F, full_matrices=False)

# print the results
print(f'U:\n {U} \n')
print(f'Sigma:\n {np.diag(Sigma)} \n')
print(f'VT:\n {VT} \n')

# create zero matrix and populate with the three largest values
Sigma_reduced = np.zeros((U.shape[0], VT.shape[0]))
Sigma_reduced[:3, :3] = np.diag(Sigma[:3])
print(Sigma_reduced)


# reconstruct F using the three largest values
F_reconstructed = np.dot(U, np.dot(Sigma_reduced, VT))
print(f'F reconstructed:\n{F_reconstructed}')

# show 2d representations of keywords and documents

# keywords
U_2D = U[:, :2]

# documents
V_2D = VT[:2, :].T

print(f'2D representation of keywords:\n{U_2D}')
print(f'2D representation of documents:\n{V_2D}')


# 2d representations of keywords and documents

plt.scatter(U_2D[:, 0], U_2D[:, 1], color='blue', label="Keywords")
for i, txt in enumerate(['K1', 'K2', 'K3', 'K4', 'K5']):
    plt.annotate(txt, (U_2D[i, 0], U_2D[i, 1]))

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("2D Representation of Keywords")
plt.legend()
plt.show()

plt.scatter(V_2D[:, 0], V_2D[:, 1], color='red', label="Documents")
for i, txt in enumerate(['D1', 'D2', 'D3', 'D4', 'D5', 'D6']):
    plt.annotate(txt, (V_2D[i, 0], V_2D[i, 1]))

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("2D Representation of Documents")
plt.legend()
plt.show()

# Calculate the Document Similarity matrix using cosine similarity
similarity_matrix = cosine_similarity(F)
print(f'Document Similarity Matrix:\n{similarity_matrix}')

