# LINMA2380 Homework 2
# Group 3

import numpy as np
import matplotlib.pyplot as plt

# Question E1
def f(xsi) : 
    if xsi < 0 : 
        ValueError("xsi must be greater than 0")
    if xsi > 1 : 
        ValueError("xsi must be less than 1")
    
    if xsi <= 0.2 :
        return -1
    elif xsi >= 0.8 :
        return 2
    else :
        return 5*xsi - 2
    
n = 101
h= 1/(n-1)

MatA = np.zeros((n,n))
for i in range(n) :
    MatA[i,i] = -2
    if i > 0 :
        MatA[i,i-1] = 1
    if i < n-1 :
        MatA[i,i+1] = 1
MatA[0,0] = 1
MatA[n-1,n-1] = 1

A = (1/h**2) * MatA

b = np.zeros(n)
for i in range(n) :
    b[i] = f(i*h)
b[0] = 0
b[n-1] = 0




