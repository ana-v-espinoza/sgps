"""
Find the maximum eigenvalue in absolute value using the power method
"""

import numpy as np
from scipy.sparse import *
from time import time
from matplotlib import pyplot as plt

def powerMethod(A,x,N):
    """
    Power method on A with an initial eigenvector guess of x for N interations
    """

    y = np.zeros_like(x)
    lamb = np.zeros(N)

    for i in range(0,N):
        y = A.dot(x)
        x = y/np.linalg.norm(y)
        tmp = A.dot(x)
        #print(np.dot(x,tmp))
        lamb[i] = np.dot(x,tmp)

    return lamb

def createModelMatrix(dX):
    """
    Create a matrix which represents the discrete Poisson Equation in 1D
    assuming Dirichlet Boundary Condiitons. Can easily be extended to include
    mixed BCs
    """

    # There is 1 more grid point than the number of spaces between grid points
    dim = dX.size+1

    # The number of elements in the tridiagonal matrix, remembering that the
    # first and last columns only have 2 elements
    elements = 3*(dim-2)+4

    rows = np.zeros(elements)
    columns = np.zeros(elements)
    data = np.zeros(elements)

    # We ignore Neumann BCs for now, cause I'm lazy
    rows[0] = 0
    columns[0] = 0
    data[0] = -2/(dX[0]*dX[1])

    rows[1] = 0
    columns[1] = 1
    data[1] = 2/(dX[1]*(dX[0]+dX[1]))

    rows[-1] = (dim-1)
    columns[-1] = (dim-1)
    data[-1] = -2/(dX[-1]*dX[-2])

    rows[-2] = (dim-1)
    columns[-2] = (dim-2)
    data[-2] = 2/(dX[-2]*(dX[-1]+dX[-2]))

    # Interior points
    for i in range(1,dim-1):
       rows[3*i-1] = i
       rows[3*i] = i
       rows[3*i+1] = i

       columns[3*i-1] = i-1
       columns[3*i] = i
       columns[3*i+1] = i+1

       data[3*i-1] = 2/(dX[i-1]*(dX[i-1]+dX[i]))
       data[3*i] = -2/(dX[i-1]*dX[i])
       data[3*i+1] = 2/(dX[i]*(dX[i-1]+dX[i]))

    # Create sparse matrix using scipy functions
    A = coo_matrix((data,(rows,columns)),shape=(dim,dim))
    A = A.tocsr()
    return A

def jacobify(A):
    """
    Decompose A into an upper, lower, and diagonal matrix in order to find the
    iteration matrix for Jacobi relaxation
    """

    L = tril(A,k=-1,format="csr")
    U = triu(A,k=1,format="csr")
    D = A-L-U

    diagonals = A.diagonal()
    Dinv = diags(diagonals**-1,format="csr")

    T = -Dinv*(L+U)

    # Find eigenvalues of T by first finding eigenvalues of -Dinv*A


    return T

for j in range(3,8):
    dX = np.ones(j)

    A = createModelMatrix(dX)
    T = jacobify(A)

    x = np.random.rand(dX.size+1)

    powerIter = 2000
    lamb = np.zeros(powerIter)
    for i in range(0,powerIter):
        x = np.random.rand(dX.size+1)
        #print(x)
        LAMB = powerMethod(T,x,200)
        lamb[i] = LAMB[-1]
    print("N = {}".format(j))
    print(lamb[-1])
    print(np.mean(lamb), np.std(lamb))
    print()

# Checking to see if I actually know how to do x^T A x
#y = np.array([1,2])
#B = np.array([[1,2],[3,4]])
#B = csr_matrix(B)
#print(y)
#print(B)
#tmp = B.dot(y)
#print(tmp)
#print(y.dot(tmp))
