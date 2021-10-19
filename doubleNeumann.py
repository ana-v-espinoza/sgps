from sgps import StretchedGridPoisson
import numpy as np
from matplotlib import pyplot as plt

"""
Demonstrate that the spectral radius of a 1D model matrix with Neumann BCs at
both boundaries will not be < 1
"""

nx = 100
x = np.arange(0,nx)

#############################################################
# Double Neumann BCs
#############################################################

sgps = StretchedGridPoisson(1,X1=x)

Ax = sgps.set_1DA("X1", lbcNeumann=True, ubcNeumann=True)
A,T,rhoT = sgps.set_modelMatrix()
sgps.set_forcing(np.zeros_like(x))

# Manually apply jacobi's method, as the class method sgps.jacobisMethod will
# determine you need infinite iterations to reduce the error (because the
# spectral radius is >= 1). Note the zero forcing.

t = 200000
phi = 0*np.ones_like(x)
for i in range(0,int(t)):
    phi = T*phi

plt.plot(x,phi)
plt.xlabel("X")
plt.ylabel("Soln")
plt.title("phi0 = 0; t = 200000")
plt.tight_layout(rect=[0,0.05,0.95,0.95])
plt.savefig("./doubleNeumann0.png")
plt.show()
plt.close()

t = 200000
phi = 50*np.ones_like(x)
for i in range(0,int(t)):
    phi = T*phi

plt.plot(x,phi)
plt.xlabel("X")
plt.ylabel("Soln")
plt.title("phi0 = 50; t = 200000")
plt.tight_layout(rect=[0,0.05,0.95,0.95])
plt.savefig("./doubleNeumann50.png")
plt.show()
plt.close()

#############################################################
# UNBC
#############################################################

sgps = StretchedGridPoisson(1,X1=x)

Ax = sgps.set_1DA("X1", ubcNeumann=True)
A,T,rhoT = sgps.set_modelMatrix()
# sgps.set_boundaryValues("X1", "lower", nx-1)
f = sgps.set_forcing(np.zeros_like(x))
f = A.diagonal()**(-1)*f

# Manually apply jacobi's method, as the class method sgps.jacobisMethod will
# determine you need infinite iterations to reduce the error (because the
# spectral radius is >= 1).

t = 200000
phi = 50*np.ones_like(x[1:])
for i in range(0,int(t)):
    phi = T*phi+f

plt.plot(x[1:],phi)
plt.xlabel("X")
plt.ylabel("Soln")
plt.title("phi0 = 0; t = 200000")
plt.tight_layout(rect=[0,0.05,0.95,0.95])
plt.savefig("./doubleNeumannLDBC0.png")
plt.show()
plt.close()

t = 200000
phi = 50*np.ones_like(x[1:])
for i in range(0,int(t)):
    phi = T*phi+f

plt.plot(x[1:],phi)
plt.xlabel("X")
plt.ylabel("Soln")
plt.title("phi0 = 50; t = 200000")
plt.tight_layout(rect=[0,0.05,0.95,0.95])
plt.savefig("./doubleNeumannLDBC50.png")
plt.show()
plt.close()
