import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse.linalg import gmres
# LINMA2380 Homework 2

# E1
def f(x):
  if x <= 0.2:
    return -1
  elif x < 0.8:
    return 5*x-2
  else:
    return 2

class KrylovSolver:
  def __init__(self, n, r):
    self.n = n
    self.h = 1/(n-1)
    self.r = r
    self.generate_Ab()
    self.arnoldi_method()

  def generate_Ab(self):
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
    self.A = ( 1/self.h ** 2 ) * A
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

n = 101
rs = [10, 20, 30, 40, 50, 70, 100]

plt.figure(figsize=(10, 6))

for r in rs:
    solver = KrylovSolver(n, r)
    u = solver.solve() 
    #u, _ = gmres(solver.A, solver.b, restart=None, atol=1e-12)  # Use 'atol' for tolerance
    x = np.linspace(0, 1, n)
    plt.plot(x, u, label=f'r = {r}')

plt.title('Solution of the 1D Poisson Equation for Different Krylov Subspace Dimensions')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid(True)
plt.show()



# E2
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

# find eigenvaues of A and associated eigenvectors
  def d2(self):
    eigvals, eigvecs = np.linalg.eig(self.A)
    eigvecs = eigvecs.T
    return eigvals, eigvecs

# Plot the eigenvalues of A and the associated eigenvectors
  def plot_eigenvalues(self):
    eigvals, eigvecs = self.d2()
    plt.figure(figsize=(10, 6))
    plt.plot(eigvals, 'o')
    plt.title('Eigenvalues of A')
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    for i in range(5):
      plt.plot(eigvecs[i], label=f'Eigenvalue {i}')
    plt.title('Eigenvectors of A')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plotting the 5 first eigenvectors of H
  def plot_eigenvectors_of_H(self):
    plt.figure(figsize=(10, 6))
    for i in range(5):
       plt.plot(self.Q[:,i], label=f'Eigenvector {i}')
    plt.title('Eigenvectors of H')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

  # def solve(self):
  #   y = np.linalg.lstsq(self.H, self.Q.T @ self.b, rcond=None)[0]
  #   return self.Q @ y


KrylovSolver.plot_eigenvalues(KrylovSolver(101, 50))

# using the command to show the eigenvectors of H
KrylovSolver.plot_eigenvectors_of_H(KrylovSolver(101, 50))

