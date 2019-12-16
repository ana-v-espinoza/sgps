import numpy as np

stretchedGrid = True
stretchedName = '_stretchedGrid_' if stretchedGrid else '_evenGrid_'
numOfIt = 4000
filename = './output/parPlates' + stretchedName + 'iter' + str(numOfIt) + '.npz'

# Defining the domain, can be uneven
# The x, y, and z values defined below are grid centers
dx = 1000 # m
dy = 1000 # m

nx = 100
ny = 100
nz = 86 #100

x = dx*np.arange(0,nx)
y = dy*np.arange(0,ny)

# Stretched grid to match Shelby's stretched grid
if stretchedGrid:
    z = np.array(
    [   50.0,  153.94739 ,   265.7895  ,   385.52637,    513.15796,
       648.68427,    792.10535,    943.42114,   1102.6316,    1269.7368,
      1444.7369,    1627.6317,    1818.4211,    2017.1055,    2223.6843,
      2438.1582,    2660.5264,    2890.7896,    3128.9475,    3375.,
      3625. ,       3875. ,       4125. ,       4375. ,       4625.,
      4875. ,       5125. ,       5375. ,       5625. ,       5875.,
      6125. ,       6375. ,       6625. ,       6875. ,       7125.,
      7375. ,       7625. ,       7875. ,       8125. ,       8375.,
      8625. ,       8875. ,       9125. ,       9375. ,       9625.,
      9875. ,      10125. ,      10375. ,      10625. ,      10875.,
     11125. ,      11375. ,      11625. ,      11875. ,      12125.,
     12375. ,      12625. ,      12875. ,      13125. ,      13375.,
     13625. ,      13875. ,      14125. ,      14375. ,      14625.,
     14875. ,      15125. ,      15375. ,      15625. ,      15875.,
     16125. ,      16375. ,      16625. ,      16875. ,      17125.,
     17375. ,      17625. ,      17875. ,      18125. ,      18375.,
     18625. ,      18875. ,      19125. ,      19375. ,      19625.,
     19875. ]
    )/10000.
else:
    z = np.arange(50,19625,(19625-50)/nz)/10000

# print("Grid shape:")
# print((x.shape,y.shape,z.shape))

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

# Reshape the X, Y, and Z coefficients so that elementwise
# multiplication is done correctly in the iterative solver.
# For more information see the python docs on "broadcasting"

# Reshape all the x terms
A = A[:,np.newaxis,np.newaxis];
B = B[:,np.newaxis,np.newaxis];
C = C[:,np.newaxis,np.newaxis];

# Reshape all the y terms
D = D[np.newaxis,:,np.newaxis];
E = E[np.newaxis,:,np.newaxis];
F = F[np.newaxis,:,np.newaxis];

# Reshape all the z terms
G = G[np.newaxis,np.newaxis,:];
H = H[np.newaxis,np.newaxis,:];
J = J[np.newaxis,np.newaxis,:];

# Define a forcing function, in this example it's zeros
# in the entire domain
f = np.zeros((nx,ny,nz))

# Take an initial guess at the solution, in this example
# it's zeros in the entire domain
phi = np.zeros((nx,ny,nz))

# Apply any Dirichlet boundary conditions. In this example
# we're setting the bottom boundary to be 100 (unitless)
phi[:,:,0] = 100*np.ones((nx,ny))

# Make a copy of phi to save the initial state later
phiInit = phi.copy()

# print("Coefficient shapes")
# print([[A.shape, B.shape, C.shape],
#     [D.shape, E.shape, F.shape],
#     [G.shape, H.shape, J.shape]])
# 
# print("X term shapes")
# print(phi[:-2,1:-1,1:-1].shape)
# # print((A*phi[:-2,1:-1,1:-1]).shape)
# 
# print("Z term shapes")
# print(phi[1:-1,1:-1,:-2].shape)
# print((G*phi[1:-1,1:-1,:-2]).shape)

# Begin iterative solver
for i in range(0,numOfIt):

    if i % 200 == 0:
        print("Iteration: ", i)

    # Calculate boundary conditions if they are Neumman
    # Not done here as we're assuming Dirichlet BCs

    # Apply the method of relaxation
    phi[1:-1,1:-1,1:-1] = 1./(B+E+H)*(
            f[1:-1,1:-1,1:-1]
            - A*phi[:-2,1:-1,1:-1] - C*phi[2:,1:-1,1:-1]
            - D*phi[1:-1,:-2,1:-1] - F*phi[1:-1,2:,1:-1]
            - G*phi[1:-1,1:-1,:-2] - J*phi[1:-1,1:-1,2:])

# Save the final result and initial guess in a file
# for plotting
np.savez_compressed(filename, phiInit=phiInit, phi=phi)
