import numpy as np

# Generate the identity matrix
A = np.eye(3)
B = np.eye(3)

# Association matrices
# vstack - vertically
# hstack - horizontally
AB = np.vstack((A, B))
print(AB)
