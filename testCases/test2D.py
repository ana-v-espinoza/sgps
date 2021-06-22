import numpy as np
from sgpsKron import *
from matplotlib import pyplot as plt
from genPlots2D import genPlots

# Test the solver on a parallel plates problem with Dirchlet Boundary Conditions
# at both the lower and upper boundaries in the x dimension and zero forcing
# function. The y dimension is the "long" dimension

# Zero forcing function means the solution is defined by the boundary
# conditions and should be a straight line:
def DBCS(X,Y,phiBot,phiTop):
    return (phiTop - phiBot)*X/X[:,-1] + phiBot

def LNBC(X,Y,phiPBot,phiTop):
    return phiPBot*(X-X[:,-1])+phiTop

def UNBC(X,Y,phiBot,phiPTop):
    return phiPTop*X+phiBot


dims = 1
plots = 0
const = 2000
decPlaces = 6

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

########################################
# Linear Grid
########################################
gridType = "lin"
dX = (xL-x0)/(nX-1) # m
X = iX*dX

phiTop = 100
phiBot = 0
phiPBot = (phiTop-phiBot)/X[-1]
phiPTop = (phiTop-phiBot)/X[-1]
bcValues = np.array([[phiBot,phiTop],[phiPBot,phiPTop]])

sgps = StretchedGridPoisson(2,X1=X,X2=Y)

print("2D")
print("LINEAR GRID")
print("Dirichlet")
genPlots(sgps,DBCS,"dirichlet",(bcValues[0,0],bcValues[0,1]),decPlaces,const,gridType)
print("LNBC")
genPlots(sgps,LNBC,"LNBC",(bcValues[1,0],bcValues[0,1]),decPlaces,const,gridType)
print("UNBC")
genPlots(sgps,UNBC,"UNBC",(bcValues[0,0],bcValues[1,1]),decPlaces,const,gridType)

########################################
# Exponential Grid
########################################
gridType = "exp"
X = (xL+1)**(iX/(nX-1))-1

phiTop = 100
phiBot = 0
phiPBot = (phiTop-phiBot)/X[-1]
phiPTop = (phiTop-phiBot)/X[-1]
bcValues = np.array([[phiBot,phiTop],[phiPBot,phiPTop]])

sgps = StretchedGridPoisson(2,X1=X,X2=Y)

print("2D")
print("EXPONENTIAL GRID")
print("Dirichlet")
genPlots(sgps,DBCS,"dirichlet",(bcValues[0,0],bcValues[0,1]),decPlaces,const,gridType)
print("LNBC")
genPlots(sgps,LNBC,"LNBC",(bcValues[1,0],bcValues[0,1]),decPlaces,const,gridType)
print("UNBC")
genPlots(sgps,UNBC,"UNBC",(bcValues[0,0],bcValues[1,1]),decPlaces,const,gridType)

