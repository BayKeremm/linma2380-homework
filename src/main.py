from hw2_e2 import KrylovSolver
import numpy as np


n = 101
r = 50
solver = KrylovSolver(n, r)


A = solver.A
Q = solver.Q
H = solver.H
K_r = solver.subspace

print("A shape: ",A.shape)
print("Q shape: ", Q.shape)
print("H shape: ", H.shape)

# find beta and q 
vec = A @ Q - Q @ H # = \beta q e_{r,r}^T, where \beta is a scalar, q in R^n


e_rr = np.zeros((r,1))
e_rr[r-1] = 1

non_zero_col = np.where(np.linalg.norm(vec, axis=0) > 1e-8)[0][0]  # find first non-zero column
q = vec[:, non_zero_col] / np.linalg.norm(vec[:, non_zero_col])
q = q.reshape((q.shape[0], 1)) 

beta = np.dot(vec[:, non_zero_col], q).item()   # Since q is normalized and we want beta
                                                #   [a;b] / [c;d] = [beta;beta]
                                                #   [a,b] @ [c;d] = ac + bd 
                                                #   we know a = c * beta and b = d * beta
                                                #   c^2 beta + d^2 beta = beta since [c;d] is normalized
#print([vec[:,non_zero_col][i] / q[i] for i in range(5)])

print("q shape: ",q.shape)
print("e_rr shape: ", e_rr.shape)
print("beta : ", beta)

# Compute the expected matrix (beta * q * e_rr^T)
expected_matrix = beta * q @ e_rr.T

# Assert that `expected_matrix` is close to `vec`
np.testing.assert_almost_equal(expected_matrix, vec, decimal=8)

## Isometry: The check fails, it does give diagonal 1s but some entries are to the first decimal place
## Extend Q by adding q as a new column
##Q_extended = np.hstack([Q, q])
##expected_res = A @ Q_extended

##identity_check = Q_extended.T @ Q_extended
##np.testing.assert_almost_equal(identity_check, np.eye(Q_extended.shape[1]), decimal=8)

# Eigenpair approximation for H
eigenvalues, eigenvectors = np.linalg.eigh(H)
sorted_indices = np.argsort(eigenvalues)[::-1]
largest_eigenvalue = eigenvalues[sorted_indices[0]]
approx_eigenvector = eigenvectors[:, sorted_indices[0]]

# Compute approximate eigenvector and verify Ax - λx condition
x = Q @ approx_eigenvector  # Q is the orthonormal basis

# Final condition check in D1
lhs = np.linalg.norm(A @ x - largest_eigenvalue * x)
rhs = abs(beta * e_rr.T @ approx_eigenvector)

print(f"Left-hand side: {lhs}")
print(f"Right-hand side: {rhs}")