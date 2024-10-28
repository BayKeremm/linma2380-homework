import unittest
import numpy as np
from hw2 import KrylovSolver

import unittest
import numpy as np
from scipy.sparse.linalg import gmres

def f(x):
    if x <= 0.2:
        return -1
    elif x < 0.8:
        return 5*x-2
    else:
        return 2

class TestKrylovSolver(unittest.TestCase):
    
    def test_generate_Ab(self):
        # Test if A and b are generated correctly for a simple case
        n = 5
        r = 3
        solver = KrylovSolver(n, r)
        
        # Expected matrix A for n=5
        expected_A = 1/ solver.h ** 2 * np.array([[1, 0, 0, 0, 0],
                               [1, -2, 1, 0, 0],
                               [0, 1, -2, 1, 0],
                               [0, 0, 1, -2, 1],
                               [0, 0, 0, 0, 1]])
        
        # Check if A is correct
        np.testing.assert_array_equal(solver.A, expected_A)
        
        # Check the boundary values of b
        self.assertEqual(solver.b[0], 0)
        self.assertEqual(solver.b[-1], 0)
        
        # Test interior values of b with function f
        for i in range(1, n-1):
            np.testing.assert_almost_equal(solver.b[i], f(i * solver.h), decimal=5)

    def test_arnoldi_method(self):
        # Test if Arnoldi method generates an orthogonal basis
        n = 5
        r = 3
        solver = KrylovSolver(n, r)
        
        # Check orthogonality of Q columns
        Q = solver.Q
        for i in range(Q.shape[1]):
            for j in range(i):
                # Q[:, i] should be orthogonal to Q[:, j]
                self.assertAlmostEqual(np.dot(Q[:, i], Q[:, j]), 0, places=10)
    
    def test_h_matrix(self):
        # Test if the Hessenberg matrix h is generated correctly
        n = 5
        r = 4
        solver = KrylovSolver(n, r)
        
        H = solver.H
        # Check the dimensions of h
        self.assertEqual(H.shape, (r, r))
        calc_H = solver.Q.T @ solver.A @ solver.Q
        for i in range( H.shape[0]):
            for j in range(H.shape[1]):
                self.assertAlmostEqual(H[i, j], calc_H[i,j], places=5)
        
        # Check if h is upper Hessenberg (values below diagonal should be near zero)
        for i in range(1, H.shape[0]):
            for j in range(i-1):
                self.assertAlmostEqual(H[i, j], 0, places=5)

if __name__ == '__main__':
    unittest.main()
