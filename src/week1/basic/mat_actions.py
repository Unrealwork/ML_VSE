import numpy as np

# Create matrix manually
Z = np.array([[4, 5, 0],
              [1, 9, 3],
              [5, 1, 1],
              [3, 3, 3],
              [9, 9, 9],
              [4, 7, 1]])
# Calc sum of
r = np.sum(Z, axis=1)
indecsies = np.nonzero(r > 10)
print(indecsies)
