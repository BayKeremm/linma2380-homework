import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres

class KrylovSolver:
  def __init__(self, n, r):
    self.n = n
    self.h = 4/(n-1)
    self.r = r
    self.generate_Ab()
    self.A = self.A  + np.diag(self.v)
    self.b = np.array([i for i in range(1,n+1)])
    self.arnoldi_method()
    self.subspace = np.array([(self.A**(r)@self.b).T for i in range(self.r)])


  def generate_Ab(self):
    def V(x):
      if x <= -1:
        return 10
      elif x < 1:
        return 0
      else:
        return 10
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




if __name__=="__main__":
    n = 101
    r = 50

    plt.figure(figsize=(10, 6))

    solver = KrylovSolver(n, r)
    u_my = solver.solve() 
    u, _ = gmres(solver.A, solver.b, restart=None, atol=1e-12)
    x = np.linspace(0, 1, n)

    #plt.plot(x, u, color='royalblue', linestyle='-', linewidth=2, marker='o', markersize=4, label=r'$u$ (GMRES solution)')
    plt.plot(x, u_my, color='orangered', linestyle='--', linewidth=2, marker='s', markersize=4, label=r'$u_{\text{my}}$ (Custom Solver)')

    plt.title(f'Solution of the 1D Schrödinger Equation for r = {r}\nUsing Krylov Subspace Method', fontsize=14, fontweight='bold')
    plt.xlabel('Position $x$', fontsize=12)
    plt.ylabel('Solution $u(x)$', fontsize=12)

    #plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.show()