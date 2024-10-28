import numpy as np
class KrylovSolver:
  def __init__(self, n, r):
    self.n = n
    self.h = 4/(n-1)
    self.r = r
    self.generate_Ab()
    self.arnoldi_method()
    self.subspace = np.array([(self.A**(r)@self.b).T for i in range(self.r)])
    print(self.A)

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

    v[0] = 0
    v[self.n-1] = 0

    for i in range(1,self.n-1):
        v[i] = V(i*self.h)
    self.A = ( 1/(self.h ** 2) ) * A
    self.b = v

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
    