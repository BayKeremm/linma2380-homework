import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse.linalg import gmres
# LINMA2380 Homework 2


class KrylovSolver:
  def __init__(self, n, r):
    self.n = n
    self.h = 1/(n-1)
    self.r = r
    self.generate_Ab()
    self.arnoldi_method()
    self.subspace = np.array([(self.A**(r)@self.b).T for i in range(self.r)])

  def generate_Ab(self):
    def f(x):
      if x <= 0.2:
        return -1
      elif x < 0.8:
        return 5*x-2
      else:
        return 2
    A = np.zeros((self.n,self.n))
    A[0,0] = 1
    A[self.n-1,self.n-1] = 1

    j = 0
    for i in range(1,self.n-1):
      A[i,:][j] = 1
      A[i,:][j+1] = -2
      A[i,:][j+2] = 1
      j +=1
    
    b = np.zeros(self.n)

    b[0] = 0
    b[self.n-1] = 0

    for i in range(1,self.n-1):
      b[i] = f(i*self.h)
    self.A = ( 1/(self.h ** 2) ) * A
    self.b = b

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
    
    

# Now solve for r = 10,...,50: min_{x in Krylov subspace of r (A,b)} ||Ax-b||
# Finite difference method: https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf

if __name__ == "__main__":
  n = 101
  rs = [10, 20, 30, 40, 50]

  plt.figure(figsize=(10, 6))

  for r in rs:
      solver = KrylovSolver(n, r)
      u = solver.solve() 
      #u, _ = gmres(solver.A, solver.b, restart=None, atol=1e-12)  # Use 'atol' for tolerance
      x = np.linspace(0, 1, n)
      plt.plot(x, u, label=f'r = {r}')

  plt.title('Solution of the 1D Poisson Equation for Different Krylov Subspace Dimensions')
  plt.xlabel('Position $x$', fontsize=12)
  plt.ylabel('Solution $u(x)$', fontsize=12)
  plt.legend()
  plt.show()