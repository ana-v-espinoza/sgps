import numpy as np
from sgpsKron import *
from matplotlib import pyplot as plt
from genPlots3D import genPlots

# Test the solver on a parallel plates problem with Dirchlet Boundary Conditions
# at both the lower and upper boundaries in the x dimension and zero forcing
# function. The y dimension is the "long" dimension

# Zero forcing function means the solution is defined by the boundary
# conditions and should be a straight line:
def DBCS(X,Y,Z,phiBot,phiTop):
    return (phiTop - phiBot)*X/X[:,-1] + phiBot

def LNBC(X,Y,Z,phiPBot,phiTop):
    return phiPBot*(X-X[:,-1])+phiTop

def UNBC(X,Y,Z,phiBot,phiPTop):
    return phiPTop*X+phiBot


dims = 1
plots = 0
const = 2000
decPlaces = 2

x0 = 0
xL = 200
nX = 50
iX = np.arange(0,nX)

y0 = 0
yJ = 20000
nY = 50
iY = np.arange(0,nY)
dY = (yJ-y0)/(nY-1)
Y = iY*dY

z0 = 0
zK = 20000
nZ = 50
iZ = np.arange(0,nZ)
dZ = (zK-z0)/(nZ-1)
Z = iZ*dZ
########################################
# Linear Grid
########################################
dX = (xL-x0)/(nX-1) # m
X = iX*dX

phiTop = 100
phiBot = 0
phiPBot = (phiTop-phiBot)/X[-1]
phiPTop = (phiTop-phiBot)/X[-1]
bcValues = np.array([[phiBot,phiTop],[phiPBot,phiPTop]])

sgps = StretchedGridPoisson(3,X1=X,X2=Y,X3=Z)

print("Dirichlet \n")
genPlots(sgps,DBCS,"dirichlet",(bcValues[0,0],bcValues[0,1]),decPlaces,const)
#print("LNBC \n")
#genPlots(sgps,LNBC,"LNBC",(bcValues[1,0],bcValues[0,1]),decPlaces,const)
#print("UNBC \n")
#genPlots(sgps,UNBC,"UNBC",(bcValues[0,0],bcValues[1,1]),decPlaces,const)

########################################
# Exponential Grid
########################################

X = (xL+1)**(iX/(nX-1))-1

phiTop = 100
phiBot = 0
phiPBot = (phiTop-phiBot)/X[-1]
phiPTop = (phiTop-phiBot)/X[-1]
bcValues = np.array([[phiBot,phiTop],[phiPBot,phiPTop]])

#sgps = StretchedGridPoisson(3,X1=X,X2=Y,X3=Z)
#
#print("Dirichlet \n")
#genPlots(sgps,DBCS,"dirichlet",(bcValues[0,0],bcValues[0,1]),decPlaces,const)
#print("LNBC \n")
#genPlots(sgps,LNBC,"LNBC",(bcValues[1,0],bcValues[0,1]),decPlaces,const)
#print("UNBC \n")
#genPlots(sgps,UNBC,"UNBC",(bcValues[0,0],bcValues[1,1]),decPlaces,const)
#
