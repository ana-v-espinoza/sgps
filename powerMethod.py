"""
Find the maximum eigenvalue in absolute value using the power method
"""

import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import norm
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

def gelfandFormula(A,N):
    """
    Apply Gelfand's Forumula to approximate the spectral radius using the p=1
    matrix norm (see p-norm for matrices)

    p(A) ~ norm(A^N)^(1/N)
    """

    size = A.shape[0]
    B = identity(size)
    p = np.zeros(N)
    n = np.zeros(N)

    for i in range(0,N):
        B = B*A
        n[i] = np.linalg.norm(B.toarray(),ord='fro')
        p[i] = np.linalg.norm(B.toarray(),ord='fro')**(1./N)

    return p,n


def gelfandsFormula(A,N=200):
    """
    Apply Gelfand's Forumula to approximate the spectral radius using the p=1
    matrix norm (see p-norm for matrices)

    p(A) ~ norm(A^N)^(1/N)

    Try N=5000 iterations; if p(A) > 1 after N iterations, double the amount
    of iterations.
    """

    print("In gelfandsFormula")

    size = A.shape[0]
    B = identity(size)

    i = 2
    its = N
    p = 1
    while i < its:
        if i**2 < its:
            print(i)
            B = B.power(2)
            i = i**2
        else:
            B = B.dot(A)
            i = i+1
        p = norm(B,ord=1)**(1./i)
        if i%10 == 0:
            print(i)
        if i == its and p >= 1:
            its = 2*its
            print(its)

    return p


def createModelMatrix(dX):
    """
    Create a matrix which represents the discrete Poisson Equation in 1D
    assuming Dirichlet Boundary Condiitons. Can easily be extended to include
    mixed BCs
    """

    # There is 1 more grid point than the number of spaces between grid points
    # For DBCs, 2 of these grid points are boundary points, and so we have a
    # total matrix size of dX.size-1
    dim = dX.size-1

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

    #L = tril(A,k=-1,format="csr")
    #U = triu(A,k=1,format="csr")
    #D = A-L-U

    #diagonals = A.diagonal()
    #Dinv = diags(diagonals**-1,format="csr")

    #T = -Dinv*(L+U)

    ## Find eigenvalues of T by first finding eigenvalues of -Dinv*A


    #return T
    L = A.get_shape()
    L = L[0]
    I = identity(L)
    D = A.diagonal()
    Dinv = diags(D**(-1), format="csr")
    preT = -Dinv*A
    T = I+preT
    return preT, T

# L interior points for a total of L+2 grid points
L = 10
# L+1 dX points
dX = np.ones(L+1)

A = createModelMatrix(dX)
_,T = jacobify(A)

N = 2**16
lamb = gelfandsFormula(T)
print("L = {}".format(L))
print("Gelfand's Formula 2: p = {}".format(lamb))
p,n = gelfandFormula(T,N)

print("Gelfand's Formula: p = {}".format(p[-1]))

#powerIter = 100
#M = 2000
#lamb = np.zeros(powerIter)
#for i in range(0,powerIter):
#    x = np.random.rand(dX.size-1)
#    #print(x)
#    LAMB = powerMethod(T,x,M)
#    lamb[i] = LAMB[-1]
#print(lamb[-1])
#print(np.mean(lamb), np.std(lamb))
#print()
#
plt.plot(p,'.')
plt.plot(n,'+')
plt.show()
plt.close()
