from hw2_e1 import KrylovSolver
import numpy as np


n = 6
r = 5
solver = KrylovSolver(n, r)


A = solver.A
Q = solver.Q
H = solver.H
K_r = solver.subspace

print("A shape: ",A.shape)
print("Q shape: ", Q.shape)
print("H shape: ", H.shape)

np.testing.assert_almost_equal(A @ Q[:,:r], Q[:,:r+1] @ H, decimal=8)

# q_1, q_2, q_3 form a basis for [b, Ab, A^2b]

Q_r = Q[:,:r]
print("Q_r shape: ", Q_r.shape)
print("K_r shape: ", K_r.shape)

for i in range(r):
    x, residuals, rank, s = np.linalg.lstsq(Q_r,K_r[i,:],rcond=None)
    k_recon = Q_r @ x
    #print(x)
    #print(residuals)
    #print(rank)
    #print(s)
    np.testing.assert_almost_equal(k_recon, K_r[i,:], decimal=8)



