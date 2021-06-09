import numpy as np
from sgpsKron import *
from matplotlib import pyplot as plt

import testCases.doStuffAndThings

dims = 1
nX, nY, nZ = 6,5,4

def lexicoGraphicF(dims,nX,nY,nZ):
    if dims == 1:
        f = np.zeros(nX)
        for i in range(0,nX):
            f[i] = i
        return f
    if dims == 2:
        f = np.zeros((nX,nY))
        for i in range(0,nX):
            for j in range(0,nY):
                f[i,j] = i+nX*j
        return f
    if dims == 3:
        f = np.zeros((nX,nY,nZ))
        for i in range(0,nX):
            for j in range(0,nY):
                for k in range(0,nZ):
                    f[i,j,k] = i+nX*j+nX*nY*k
        return f


X = np.arange(0,nX)
Y = np.arange(0,nY)
Z = np.arange(0,nZ)

if dims == 1:
    Ax = sgps.set_1DA("X1")
    sgps.set_boundaryValues("X1","lower", 10)
    sgps.set_boundaryValues("X1","upper", 0)
    f = np.zeros(nX)
    # f = lexicoGraphicF(dims,nX,nY,nZ)


if dims == 2:
    sgps = StretchedGridPoisson(2,X1=X, X2=Y)
    Ax = sgps.set_1DA("X1")
    Ay = sgps.set_1DA("X2")
    sgps.set_boundaryValues("X1","lower", -0)
    sgps.set_boundaryValues("X1","upper", -0)
    sgps.set_boundaryValues("X2","lower", -0)
    sgps.set_boundaryValues("X2","upper", -0)
    f = np.zeros((nX,nY))

if dims == 3:
    sgps = StretchedGridPoisson(3,X1=X, X2=Y, X3=Z)
    Ax = sgps.set_1DA("X1")
    Ay = sgps.set_1DA("X2")
    Az = sgps.set_1DA("X3")
    sgps.set_boundaryValues("X1","lower", -0)
    sgps.set_boundaryValues("X2","lower", -0)
    sgps.set_boundaryValues("X3","lower", -0)
    sgps.set_boundaryValues("X1","upper", -0)
    sgps.set_boundaryValues("X2","upper", -0)
    sgps.set_boundaryValues("X3","upper", -0)
    f = np.zeros((nX,nY,nZ))

A, T, rhoT = sgps.set_modelMatrix()

f = sgps.set_forcing(f)
soln = sgps.jacobisMethod(0.01)
print(soln)
