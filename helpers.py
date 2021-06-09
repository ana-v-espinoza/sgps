"""
Some helper functions
"""

import numpy as np
from scipy.sparse import *
from time import time
from matplotlib import pyplot as plt

def to1D(i,j,k,I,J):
    """
    Take an element of a 3D grid and transform it to the nth element of a
    flattened vector.
    The ordering is x-coords --> y-coords --> z-coords, whatever that means to
    you.

    ARGS
    i,j,k: The ith, jth, and kth, element of the grid, where the iteration is
    over the x, y, and z coords, respectively
    I,J: The total number of i and j elements, respectively
    """
    return i+j*I+k*I*J

def createModelMatrix(dX, dY, dZ, I, J, K):
    """
    Create the matrix representing the discretized Poisson Equation in 3D, where
    the solution matrix has been flattened in the way described about in to1D

    We assume, for the time being, Dirichlet boundary conditions

    ARGS
    dX, dY, dZ: Arrays of the grid spacing in the respective directions
    I, J, K: Number of discrete grid points in the x, y, and z directions

    RETURNS
    NxN matrix representing the discretized Poisson Equation, where N = I*J*K
    """

    # Total number of elements
    N = I*J*K
    # Total number of interior elements
    NInterior = (I-2)*(J-2)*(K-2)
    # Total number of boundary condition elements
    NBound = N-NInterior

    # Each row of our model matrix corresponds to an interior point, and the
    # stencil for each element is a 7 point stencil
    # For fully Dirichlet boundary conditions, each grid element has a "1-point"
    # stencil

    # Total number of non-zero points in the model
    nonZero = 7*NInterior+1*NBound

    # Initialize arrays to use when creating the sparse matrix
    rows = np.zeros((nonZero))
    columns = rows.copy()
    data = rows.copy()

    # Used to iterate through the above arrays
    nzero = 0

    # Go through each "element" of the model matrix and decide whether it is
    # non-zero
    for i in range(0,I):
        for j in range(0,J):
            for k in range(0,K):
                row = to1D(i,j,k,I,J)
                # Handle Dirichlet boundary conditions
                if i == 0 or j == 0 or k == 0:
                    rows[nzero] = row
                    columns[nzero] = row
                    data[nzero] = 1
                    nzero += 1
                    continue
                if i == I-1 or j == J-1 or k == K-1:
                    rows[nzero] = row
                    columns[nzero] = row
                    data[nzero] = 1
                    nzero += 1
                    continue
                # The columns corresponding to the 7 point stencil
                center = row
                west = to1D(i-1,j,k,I,J)
                east = to1D(i+1,j,k,I,J)
                south = to1D(i,j-1,k,I,J)
                north = to1D(i,j+1,k,I,J)
                bottom = to1D(i,j,k-1,I,J)
                top = to1D(i,j,k+1,I,J)
                # The data corresponding to the 7 point stencil
                cData = -2.0/(dX[i-1]*dX[i])-2.0/(dY[j-1]*dY[j])-2.0/(dZ[k-1]*dZ[k])
                wData = 2.0/(dX[i-1]*(dX[i-1]+dX[i]))
                eData = 2.0/(dX[i]*(dX[i-1]+dX[i]))
                sData = 2.0/(dY[j-1]*(dY[j-1]+dY[j]))
                nData = 2.0/(dY[j]*(dY[j-1]+dY[j]))
                bData = 2.0/(dZ[k-1]*(dZ[k-1]+dZ[k]))
                tData = 2.0/(dZ[k]*(dZ[k-1]+dZ[k]))
                # Add this data to the corresponding arrays
                rows[nzero], columns[nzero], data[nzero] = row, center, cData
                nzero += 1
                rows[nzero], columns[nzero], data[nzero] = row, west, wData
                nzero += 1
                rows[nzero], columns[nzero], data[nzero] = row, east, eData
                nzero += 1
                rows[nzero], columns[nzero], data[nzero] = row, south, sData
                nzero += 1
                rows[nzero], columns[nzero], data[nzero] = row, north, nData
                nzero += 1
                rows[nzero], columns[nzero], data[nzero] = row, bottom, bData
                nzero += 1
                rows[nzero], columns[nzero], data[nzero] = row, top, tData
                nzero += 1

    A = coo_matrix((data,(rows,columns)),shape=(N,N))
    A = A.tocsr()
    return A

I = 100
J = 100
K = 50
dX = np.ones((I))
dY = np.ones((J))
dZ = np.ones((K))
elapsed = time()
A = createModelMatrix(dX,dY,dZ,I,J,K)
elapsed = time()-elapsed
print("Time to create model matrix (s): ", elapsed)

test = np.ones((I*J*K))

L = tril(A,k=-1,format="csr")
U = triu(A,k=1,format="csr")
D = A-L-U
diagonals = A.diagonal()
Dinv = diags(diagonals**-1,format="csr")

elapsed = time()
for i in range(0,10):
    test = (Dinv*(L+U))*test
elapsed = time()-elapsed
print("Time to perform 10 iterations (s): ", elapsed)

# print(A)
# plt.spy(A,markersize=1)
# plt.show()
