import numpy as np
from sgpsKron import *
from matplotlib import pyplot as plt
from genPlots1D import genPlots

# Test the solver on a parallel plates problem with Dirchlet Boundary
# Conditions at both the lower and upper boundaries and zero forcing function.
# Zero forcing function means the solution is defined by the boundary conditions
# and should be a straight line:
def DBCS(X,phiBot,phiTop):
    return (phiTop - phiBot)*X/X[-1] + phiBot

def LNBC(X,phiPBot,phiTop):
    return phiPBot*(X-X[-1])+phiTop

def UNBC(X,phiBot,phiPTop):
    return phiPTop*X+phiBot


dims = 1
plots = 0
const = 2000
decPlaces = 4

x0 = 0
xL = 200
nX = 50
iX = np.arange(0,nX)

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

sgps = StretchedGridPoisson(1,X1=X)

print("1D")
print("LINEAR GRID")
print("Dirichlet")
genPlots(X,sgps,DBCS,"dirichlet",(bcValues[0,0],bcValues[0,1]),decPlaces,const,gridType)
print("LNBC")
genPlots(X,sgps,LNBC,"LNBC",(bcValues[1,0],bcValues[0,1]),decPlaces,const,gridType)
print("UNBC")
genPlots(X,sgps,UNBC,"UNBC",(bcValues[0,0],bcValues[1,1]),decPlaces,const,gridType)

########################################
# Exponential Grid
########################################
gridType = "exp"
X = (xL+1)**(iX/(nX-1))-1

phiTop = 10
phiBot = 0
phiPBot = (phiTop-phiBot)/X[-1]
phiPTop = (phiTop-phiBot)/X[-1]
bcValues = np.array([[phiBot,phiTop],[phiPBot,phiPTop]])

sgps = StretchedGridPoisson(1,X1=X)

print("1D")
print("EXPONENTIAL GRID")
print("Dirichlet")
genPlots(X,sgps,DBCS,"dirichlet",(bcValues[0,0],bcValues[0,1]),decPlaces,const,gridType)
print("LNBC")
genPlots(X,sgps,LNBC,"LNBC",(bcValues[1,0],bcValues[0,1]),decPlaces,const,gridType)
print("UNBC")
genPlots(X,sgps,UNBC,"UNBC",(bcValues[0,0],bcValues[1,1]),decPlaces,const,gridType)

