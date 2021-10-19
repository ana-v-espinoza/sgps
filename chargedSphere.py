"""
Test SGPS on a uniformly charged sphere far away from any boundaries
"""

import numpy as np
from matplotlib import pyplot as plt
from sgps import *

#############################################################
# Define the domain
#############################################################

nx, ny, nz = 100, 100, 100

x0, xL = -1500., 1500.    # m
y0, yJ = -1500., 1500.    # m
z0, zK = -1500., 1500.    # m

dx = (xL-x0)/(nx) # m
dy = (yJ-y0)/(ny) # m
dz = (zK-z0)/(nz) # m

x = np.arange(x0,xL,dx) # m 
y = np.arange(y0,yJ,dy) # m
z = np.arange(z0,zK,dz) # m

# Cartesian indexed ie X[Y,X,Z]
X, Y, Z = np.meshgrid(x,y,z)

#############################################################
# Define uniformly charged sphere
#############################################################

radius = 100.       # m
xcenter = 0.        # m
ycenter = 0.        # m
zcenter = 0.        # m

# Choose Q so that the maximum potential is phiMax
phiMax = 10.
e0 = 8.8542*10**(-12)       # C^2/(V m)
V = 4./3.*np.pi*radius**3   # m^3
Q = 2*e0*V/radius**2*phiMax # C
p = Q/V

# Is less than 1 if inside the sphere
r = np.sqrt(
        np.square(X-xcenter)+
        np.square(Y-ycenter)+
        np.square(Z-zcenter))
beta = r/radius
inSphere = beta <= 1
outSphere = beta > 1

# The forcing function for nabla^2 phi = f
f = -1*(inSphere*p/e0)

# Change to 1 for plotting
if 0:
    pcm = plt.pcolormesh(X[:,:,int(nz/2)],Y[:,:,int(nz/2)],f[:,:,int(nz/2)])
    plt.colorbar(pcm)
    plt.title("X-Y Cross-Section through the origin")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout(rect=[0,0.05,0.95,0.95])
    plt.show()
    plt.close()

#############################################################
# Create exact solution for nabla^2 phi = -f
#############################################################

phiExact = np.zeros_like(f)

phiExact[inSphere] = p*radius**2/(2*e0)*(1-np.square(r[inSphere])/(3*radius**2))
phiExact[outSphere] = p*radius**3/(3*e0*r[outSphere])

# Change to 1 for plotting
if 1:
    pcm = plt.pcolormesh(X[:,:,int(nz/2)],Y[:,:,int(nz/2)],phiExact[:,:,int(nz/2)])
    plt.colorbar(pcm)
    plt.title("Cross-Section through Origin: Exact")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout(rect=[0,0.05,0.95,0.95])
    plt.show()
    plt.close()

    plt.plot(y,phiExact[:,int(nx/2),int(nz/2)])
    plt.title("Y Profile through Origin: Exact")
    plt.xlabel("Y (m)")
    plt.ylabel("Phi (V)")
    plt.tight_layout(rect=[0,0.05,0.95,0.95])
    plt.show()
    plt.close()

#############################################################
# Stretched Grid Poisson Solver
#############################################################

decPlaces = 3
sgps = StretchedGridPoisson(3,X1=x,X2=y,X3=z)

# Defaults to homogenous DBCs
sgps.set_1DA("X1")
sgps.set_1DA("X2")
sgps.set_1DA("X3")

sgps.set_forcing(f)
sgps.set_modelMatrix()
phi0 = np.zeros_like(f)

#############################################################
# Solve
#############################################################
soln = sgps.jacobisMethod(decPlaces,phi0=phi0)

#############################################################
# Plot SGPS
#############################################################

# Change to 1 for plotting
if 1:
    pcm = plt.pcolormesh(X[1:-1,1:-1,int(nz/2)],
        Y[1:-1,1:-1,int(nz/2)],
        soln[:,:,int(nz/2)])
    plt.colorbar(pcm)
    plt.title("Cross-Section through Origin: SGPS")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout(rect=[0,0.05,0.95,0.95])
    plt.show()
    plt.close()

    plt.plot(y[1:-1],phiExact[1:-1,int(nx/2),int(nz/2)])
    plt.title("Y Profile through Origin: SGPS")
    plt.xlabel("Y (m)")
    plt.ylabel("Phi (V)")
    plt.tight_layout(rect=[0,0.05,0.95,0.95])
    plt.show()
    plt.close()

#############################################################
# Plot Error
#############################################################

err0 = phi0[1:-1,1:-1,1:-1] - soln
err = phiExact[1:-1,1:-1,1:-1] - soln

nErr0 = np.norm(np.flatten(err0))
nErr = np.norm(np.flatten(err))

print(nErr0, nErr)

# Change to 1 for plotting
if 1:
    pcm = plt.pcolormesh(X[1:-1,1:-1,int(nz/2)],Y[1:-1,1:-1,int(nz/2)],err[:,:,int(nz/2)],cmap="bwr")
    plt.colorbar(pcm)
    plt.title("Cross-Section through Origin: Error")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout(rect=[0,0.05,0.95,0.95])
    plt.show()
    plt.close()

    plt.plot(y[1:-1],err[:,int(nx/2),int(nz/2)])
    plt.title("Y Profile through Origin: |Error|")
    plt.xlabel("Y (m)")
    plt.ylabel("Phi (V)")
    plt.tight_layout(rect=[0,0.05,0.95,0.95])
    plt.show()
    plt.close()
