import numpy as np
from hw2_e2 import KrylovSolver
import matplotlib.pyplot as plt

n = 101  # Dimension of A
r = 50   # Rank or dimension of H

# Assume KrylovSolver generates H and Q as required
solver = KrylovSolver(n, r)
A = solver.A       # A matrix (nxn)
H = solver.H       # H matrix (dimension r x r)
Q = solver.Q       # Q matrix (dimension n x r, orthonormal basis for Kr(A, b))

# Step 1: Compute eigenvalues and eigenvectors of H
eigenvalues, eigenvectors = np.linalg.eigh(H)

# Step 2: Sort eigenvalues in descending order and select top 5
sorted_indices = np.argsort(eigenvalues)[::-1]
top_eigenvectors = eigenvectors[:, sorted_indices[:5]]

# Step 3: Compute the approximate eigenvectors of A using Q
approx_eigenvectors = Q @ top_eigenvectors  # Shape (n, 5)

# Step 4: Plot the top 5 approximate eigenvectors of A
fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

for i in range(5):
    axes[i].plot(approx_eigenvectors[:, i], label=f'Approximate Eigenvector {i+1}')
    axes[i].set_title(f'Approximate Eigenvector {i+1} from H for A (Approx Eigenvalue = {eigenvalues[sorted_indices[i]]:.2f})')

plt.xlabel("Index")
plt.tight_layout()
plt.savefig("./approximated.pdf")

# Step 1: Compute eigenvalues and eigenvectors of A
eigenvalues, eigenvectors = np.linalg.eigh(A)

# Step 2: Sort eigenvalues in descending order and select top 5
sorted_indices = np.argsort(eigenvalues)[::-1]
top_eigenvectors = eigenvectors[:, sorted_indices[:5]]

# Step 3: Plot the top 5 eigenvectors of A
fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

for i in range(5):
    axes[i].plot(eigenvectors[:, i], label=f'Actual Eigenvector {i+1}')
    axes[i].set_title(f'Eigenvector {i+1} for A (Eigenvalue = {eigenvalues[sorted_indices[i]]:.2f})')

plt.xlabel("Index")
plt.tight_layout()
plt.savefig("./actual.pdf")