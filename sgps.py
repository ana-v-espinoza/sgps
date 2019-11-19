import numpy as np

# Defining the domain, can be uneven
dx = 1000 # m
dy = 1000 # m
dz = 1 # m

nx = 100
ny = 100
nz = 100

x = dx*np.arange(0,nx)
y = dy*np.arange(0,ny)
z = dz*np.arange(0,nz)

# Define the step by step grid spacing. This is in general
# distinct from dx, dy, and dz, and is only equal when
# we have a uniform grid

# Array sizes nx-1, ny-1, and nz-1
dX = x[1:]-x[:-1]
dY = y[1:]-y[:-1]
dZ = z[1:]-z[:-1]

# Make some definitions for some coefficients used in the
# calculations
Lx = dX[:-1]*dX[1:]*(dX[:-1]+dX[1:])
Ly = dY[:-1]*dY[1:]*(dY[:-1]+dY[1:])
Lz = dZ[:-1]*dZ[1:]*(dZ[:-1]+dZ[1:])

A = dX[1:]/Lx; B = -(dX[:-1]+dX[1:])/Lx; C = dX[:-1]/Lx
D = dY[1:]/Ly; E = -(dY[:-1]+dY[1:])/Ly; F = dY[:-1]/Ly
G = dZ[1:]/Lz; H = -(dZ[:-1]+dZ[1:])/Lz; J = dZ[:-1]/Lz

# Define a forcing function, in this example it's zeros
# in the entire domain
f = np.zeros((nx,ny,nz))

# Take an initial guess at the solution, in this example
# it's zeros in the entire domain
phi = np.zeros((nx,ny,nz))

# Begin iterative solver

numOfIt = 1000

print(G.shape)
print(phi[1:-1,1:-1,:-2].shape)
print(J.shape)
print(phi[1:-1,1:-1,2:].shape)

for i in range(0,numOfIt):

    if i%50 == 0:
        print("Iteration: ", i)

    # Calculate boundary conditions if they are Neumman
    # Not done here as we're assuming Dirichlet BCs

    phi[1:-1,1:-1,1:-1] = 1./(B+E+H)*(
            f[1:-1,1:-1,1:-1]
            - A*phi[:-2,1:-1,1:-1] - C*phi[2:,1:-1,1:-1]
            - D*phi[1:-1,:-2,1:-1] - F*phi[1:-1,2:,1:-1]
            - G*phi[1:-1,1:-1,:-2] - J*phi[1:-1,1:-1,2:])
