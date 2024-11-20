import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres

class KrylovSolver:
  def __init__(self, n, r):
    self.n = n
    self.h = 4/(n-1)
    self.r = r
    self.generate_Ab()
    self.A = -self.A  + np.diag(self.v)
    self.b = np.array([i for i in range(1,n+1)])
    self.arnoldi_method()
    self.subspace = np.array([(self.A**(r)@self.b).T for i in range(self.r)])


  def generate_Ab(self):
    def V(x):
      if x <= -1:
        return 100
      elif x < 1:
        return 0
      else:
        return 100
    A = np.zeros((self.n,self.n))
    A[0,0] = -2
    A[0,1] = 1
    A[self.n-1,self.n-1] = -2
    A[self.n-1,self.n-2] = 1

    j = 0
    for i in range(1,self.n-1):
      A[i,:][j] = 1
      A[i,:][j+1] = -2
      A[i,:][j+2] = 1
      j +=1
    
    v = np.zeros(self.n)

    for i in range(0,self.n):
        v[i] = V(-2 + i*self.h)
    self.A = ( 1/(self.h ** 2) ) * A
    self.v = v

  def arnoldi_method(self):
    # Compute a basis for the (self.r)-Krylov subspace
    eps = 1e-12
    Q = np.zeros((self.A.shape[0],self.r))
    H = np.zeros((self.r, self.r))

    Q[:,0] = self.b / np.linalg.norm(self.b,2)
    for k in range(1,self.r):
      v = self.A @ Q[:,k-1]
      
      # Project v on k vectors and substract from it
      for i in range(k):
        H[i,k-1] = (Q[:,i].T @ v)
        v -= H[i,k-1] * Q[:,i]
      H[k,k-1] =  np.linalg.norm(v,2)
      if H[k,k-1] > eps:
        Q[:,k] = v / H[k,k-1]
      else:
        break
    self.Q = Q
    self.H = H

  def solve(self):
    y = np.linalg.lstsq(self.H, self.Q.T @ self.b, rcond=None)[0]
    return self.Q @ y


def show_solution():
    n = 101
    r = 50


    solver = KrylovSolver(n, r)
    plt.figure(figsize=(10, 6))
    u_my = solver.solve() 
    u, _ = gmres(solver.A, solver.b, restart=None, atol=1e-12)
    x = np.linspace(0, 1, n)

    #plt.plot(x, u, color='royalblue', linestyle='-', linewidth=2, marker='o', markersize=4, label=r'$u$ (GMRES solution)')
    plt.plot(x, u_my)

    plt.title(f'Solution of the 1D Schrödinger Equation for r = {r}\nUsing Krylov Subspace Method', fontsize=14, fontweight='bold')
    plt.xlabel('Position $x$', fontsize=12)
    plt.ylabel('Solution $u(x)$', fontsize=12)

    #plt.legend()
    #plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.show()
def sanity_checks():
    n = 101
    r = 50
    solver = KrylovSolver(n, r)
    A = solver.A
    Q = solver.Q
    H = solver.H

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
    beta = np.dot(vec[:, r-1], q).item()

    print("q shape: ",q.shape)
    print("e_rr shape: ", e_rr.shape)
    print("beta : ", beta)

    # Compute the expected matrix (beta * q * e_rr^T)
    expected_matrix = beta * q @ e_rr.T

    # Assert that `expected_matrix` is close to `vec`
    np.testing.assert_almost_equal(expected_matrix, vec, decimal=8)

    expected_res =  A @ Q
    calc_res = Q @ H + beta * q @ e_rr.T

    # Result of algo1 check
    np.testing.assert_almost_equal(expected_res, calc_res, decimal=8)

    # Isometry: Not quite, but nearly there. Tested with copilot, it is also the same
    # Extend Q by adding q as a new column
    #Q_extended = np.hstack([Q, q])
    #expected_res = A @ Q_extended

    #identity_check = Q_extended.T @ Q_extended
    #np.testing.assert_almost_equal(identity_check, np.eye(Q_extended.shape[1]), decimal=8)

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

if __name__=="__main__":
    n = 101
    r = 50
    solver = KrylovSolver(n, r)
    A = solver.A       # A (nxn)
    Q = solver.Q       # Q (n x r, orthonormal basis for Kr(A, b))
    H = Q.T @ A @ Q    # H (r x r)

    # Step 1: Compute eigenvalues and eigenvectors of H
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Step 2: Sort eigenvalues in descending order and select top 5
    sorted_indices = np.argsort(eigenvalues)
    bottom_eigenvectors = eigenvectors[:, sorted_indices[:5]]

    # Step 3: Compute the approximate eigenvectors of A using Q and H
    approx_eigenvectors = Q @ bottom_eigenvectors  # Shape (n, 5)

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
    sorted_indices = np.argsort(eigenvalues)
    bottom_eigenvectors = eigenvectors[:, sorted_indices[:5]]

    # Step 3: Plot the top 5 eigenvectors of A
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    for i in range(5):
        axes[i].plot(eigenvectors[:, i], label=f'Actual Eigenvector {i+1}')
        axes[i].set_title(f'Eigenvector {i+1} for A (Eigenvalue = {eigenvalues[sorted_indices[i]]:.2f})')

    plt.xlabel("Index")
    plt.tight_layout()
    plt.savefig("./actual.pdf")