"""
Jacobi's method on 1, 2, or 3 dimensions using the Kronecker Product
"""

import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import norm
from matplotlib import pyplot as plt

class StretchedGridPoisson:

    """
    Work flow should be:
        1) Create class and tell it the number of dimensions of your problem
            Give the class the grid information for each dimension
        2) Give the class BC information for each dimension
            Create 1D model matrices
        3) Create full model matrix and iteration matrix
            Find spectral radius using power method
        4) Specify boundary values
        6) Specify forcing function in the interior of the domain and at the
            boundaries where Neumann conditions are specified
        7) Create full forcing vector, including boundary values
        5) Solve iteratively up to some prescribed reduction in error
    """

    def __init__(self,numOfDims,X1=None,X2=None,X3=None):
        """
        Copy arrays of grid centers and find the distances between adjacent grid
        centers

        Arrays are named X1,2,3 when taken as a kwarg so as to not trick the
        user into thinking they need to set X,Z when, for example, they want to
        solve an equation in the X,Z plane.

        Boundary values are defaulted to be homogenous

        """

        # 0th index is the lower boundary, 1st index is the upper boundary
        self.bcX = np.zeros(2)
        self.bcY = np.zeros(2)
        self.bcZ = np.zeros(2)

        # BC type is set to Dirichlet by default
        self.bcNeumannX = np.array([False, False])
        self.bcNeumannY = np.array([False, False])
        self.bcNeumannZ = np.array([False, False])

        self.dims = numOfDims
        self.Ax, self.Ay, self.Az = None, None, None
        if X1 is not None:
            self.X, self.nX = np.copy(X1), X1.size
            self.dX = X1[1:]-X1[:-1]
        else:
            self.X, self.nX = None, None
            self.dX = None

        if X2 is not None:
            self.Y, self.nY = np.copy(X2), X2.size
            self.dY = X2[1:]-X2[:-1]
        else:
            self.Y, self.nY = None, None
            self.dY = None

        if X3 is not None:
            self.Z, self.nZ = np.copy(X3), X3.size
            self.dZ = X3[1:]-X3[:-1]
        else:
            self.Z, self.nZ = None, None
            self.dZ = None

    def set_1DA(self, dimension, lbcNeumann=False, ubcNeumann=False):
        """
        Create the model matrix for the X/Y/Z coordinate given the Type (Neumann
        or Dirichlet) of the lower and upper boundary condition. Defaults to
        Dirichlet

        dimension is a string, either "X1", "X2", or "X3", depending on which 1D
        matrix you want to set
        """
        # Here, n is the number of interior grid points
        if dimension == "X1" and self.X is not None:
            n = self.nX-2
            dX = self.dX
            self.bcNeumannX = np.array([lbcNeumann,ubcNeumann])
        elif dimension == "X2" and self.Y is not None:
            n = self.nY-2
            dX = self.dY
            self.bcNeumannY = np.array([lbcNeumann,ubcNeumann])
        elif dimension == "X3" and self.Z is not None:
            n = self.nZ-2
            dX = self.dZ
            self.bcNeumannZ = np.array([lbcNeumann,ubcNeumann])
        else:
            # Not actual error handling
            print("""
            dimension must be one of the following strings: "X1", "X2", "X3".
            Also, ensure the respective grid information was given when
            initializing class object.""")
            return

        # Neumann BCs should only be set at a single boundary, or else we have
        # an inconsistent system. If NBCs prescribed at one of the boundaries,
        # the matrix will be (n+1) x (n+1) rather than n x n for DBCs at both
        # boundaries
        # n = n+1 if (lbcNeumann or ubcNeumann) else n
        n = n+lbcNeumann+ubcNeumann

        # The number of elements in the tridiagonal matrix, remembering that the
        # first and last columns only have 2 elements
        elements = 3*(n-2)+4

        rows = np.zeros(elements)
        columns = np.zeros(elements)
        data = np.zeros(elements)

        # First row of matrix, corresponding to the lower BC
        rows[0] = 0
        columns[0] = 0
        # -1 if Neumann
        data[0] = -1 if lbcNeumann else -2/(dX[0]*dX[1])

        rows[1] = 0
        columns[1] = 1
        # 1 if Neumann
        data[1] = 1 if lbcNeumann else 2/(dX[1]*(dX[0]+dX[1]))

        # Last row of matrix, corresponding to the upper BC
        rows[-1] = (n-1)
        columns[-1] = (n-1)
        # 1 if Neumann
        data[-1] = 1 if ubcNeumann else -2/(dX[-1]*dX[-2])

        rows[-2] = (n-1)
        columns[-2] = (n-2)
        # -1 if Neumann
        data[-2] = -1 if ubcNeumann else 2/(dX[-2]*(dX[-1]+dX[-2]))

        # Interior points
        for i in range(1,n-1):
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
        if dimension == "X1" and self.X is not None:
            self.Ax = coo_matrix((data,(rows,columns)),shape=(n,n))
            self.Ax = self.Ax.tocsr()
            return self.Ax
        if dimension == "X2" and self.Y is not None:
            self.Ay = coo_matrix((data,(rows,columns)),shape=(n,n))
            self.Ay = self.Ay.tocsr()
            return self.Ay
        if dimension == "X3" and self.Z is not None:
            self.Az = coo_matrix((data,(rows,columns)),shape=(n,n))
            self.Az = self.Az.tocsr()
            return self.Az

    def set_modelMatrix(self):
        """
        Create the full model matrix encompassing every direction. Also, create
        the iteration matrix and calculate its largest eigenvalue using the
        power method
        """

        # 1st dimension
        IX = identity(self.Ax.shape[0], format="csr")
        AX = self.Ax
        A = AX
        # Account for the 2nd dimension
        if self.Ay is not None:
            IY = identity(self.Ay.shape[0], format="csr")
            AX = kron(IY,AX)
            AY = kron(self.Ay,IX)
            A = AY+AX

        # Account for the 3rd dimension
        if self.Az is not None:
            IZ = identity(self.Az.shape[0], format="csr")
            AX = kron(IZ,AX)
            AY = kron(IZ,AY)
            AZ = kron(self.Az,kron(IY,IX))
            A = AZ+AY+AX

        self.A = A

        # Create iteration matrix
        preT, T = self.jacobify(A)
        self.preT = preT
        self.T = T

        # Find spectral radius of iteration matrix
        # Doing so uses the matrix preT, not the iteration matrix T, because the
        # I term (see the jacobiy method) introduces a constant offset related
        # to the initial guess, so we iterate on D^-1*A instead
        lamb = self.powerMethod(preT, N=5000)
        rhoT = np.abs(lamb[-1]+1)
        print("Power Method p: {}".format(rhoT))

        # Find the spectral radius of iteration matrix using Gelfand's Forumula,
        # which should always yield an estimate greater than or equal to the
        # actual spectral radius
        # lamb = self.gelfandsFormula(T)
        # print("Gelfand's Formula p: {}".format(lamb))
        self.rhoT = rhoT

        return A, T, rhoT

    def set_boundaryValues(self, dimension, boundary, value):
        """
        ARGS
        dimension:  The string "X1", "X2", or "X3"
        boundary:   The string "upper" or "lower" to specify which string is being
                    operated on
        value:      The constant value for the specified boundary condition,
                    must be a constant/scalar value

        Does NOT differentiate between Dirichlet or Neumann BCs, to set those
        the model matrix must be re-set using "set_1DA" for that particular
        dimension
        """

        if dimension == "X1" and boundary == "lower":
           self.bcX[0] = value
        elif dimension == "X1" and boundary == "upper":
            self.bcX[1] = value
        elif dimension == "X2" and boundary == "lower":
            self.bcY[0] = value
        elif dimension == "X2" and boundary == "upper":
            self.bcY[1] = value
        elif dimension == "X3" and boundary == "lower":
            self.bcZ[0] = value
        elif dimension == "X3" and boundary == "upper":
            self.bcZ[1] = value
        else:
            # Not actual error handling
            print("""
            dimension must be one of the following strings: "X1", "X2", "X3".
            boundary must be one of the following strings: "lower", "upper".
            """)

    def set_forcing(self, forcing):
        """
        Sets the forcing function over the entire domain and lexicographically
        orders it

        f must be an numOfDim dimensional array specifying the forcing function
        for each grid point in the domain. f must be Cartesian indexed as
        f[Y,X,Z] ( see the numpy documentation for np.meshgrid() )
        returns the forcing at the ith grid point in the x direction, jth
        gridpoint in the y direction, and kth gridpoint in the z direction

        set_1DA must be called before this function so the boundary information
        has been given to the class or else it will default to Dirichlet
        Boundary Conditions

        set_boundaryValues must be called before this function so the boundary
        values are known, else they will default to homogenous BCs

        """

        # Copy the array to ensure we're not working with a reference/window to
        # the array, then move the axis so f is indexed as f[Z,Y,X] instead in
        # the 3D case, which is necessary when applying subsequent matrix
        # operations
        fNoBCs = forcing.copy()

        if self.dims == 1:
            f = self.forcing1D(fNoBCs)
        if self.dims == 2:
            f = self.forcing2D(fNoBCs)
        if self.dims == 3:
            fNoBCs = np.moveaxis(fNoBCs,0,1)
            f = self.forcing3D(fNoBCs)

        self.f = f

        return f

    def forcing1D(self, f):

        # lbcX/Y/Z are used to create slices of f
        # if not Neumann (ie Dirichlet) then omit the endpoint
        lbcX = 1 if not self.bcNeumannX[0] else 0
        ubcX = self.nX-1 if not self.bcNeumannX[1] else self.nX

        if self.bcNeumannX[0]:
            f[lbcX] = self.bcX[0]*self.dX[0]
        else:
            f[lbcX] = f[lbcX]-2.*self.bcX[0]/(self.dX[0]*(self.dX[1]+self.dX[0]))
        if self.bcNeumannX[1]:
            f[ubcX-1] = self.bcX[1]*self.dX[-1]
        else:
            f[ubcX-1] = f[ubcX-1]-2.*self.bcX[1]/(self.dX[-1]*(self.dX[-1]+self.dX[-2]))

        F = f[lbcX:ubcX]
        return F

    def forcing2D(self, f):

        # lbcX/Y/Z are used to create slices of f
        # if not Neumann (ie Dirichlet) then omit the endpoint

        # X Boundary Conditions
        # The lower x boundary is found at all y and the first x index: f[:,iX0]
        # The upper x boundary is found at all y and the last x index: f[:,iX]
        iX0 = 1 if not self.bcNeumannX[0] else 0
        iX = self.nX-1 if not self.bcNeumannX[1] else self.nX

        if self.bcNeumannX[0]:
            f[:,iX0] = self.bcX[0]*self.dX[0]
        else:
            f[:,iX0] = f[:,iX0]-2.*self.bcX[0]/(self.dX[0]*(self.dX[1]+self.dX[0]))
        if self.bcNeumannX[1]:
            f[:,iX-1] = self.bcX[1]*self.dX[-1]
        else:
            f[:,iX-1] = f[:,iX-1]-2.*self.bcX[1]/(self.dX[-1]*(self.dX[-1]+self.dX[-2]))

        # Y Boundary Conditions
        # The lower y boundary is found at all x,z and the first y index: f[:,iY0,:]
        # The upper y boundary is found at all x,z and the last y index: f[:,iY,:]
        iY0 = 1 if not self.bcNeumannY[0] else 0
        iY = self.nY-1 if not self.bcNeumannY[1] else self.nY

        if self.bcNeumannY[0]:
            f[iY0,:] = self.bcY[0]*self.dY[0]
        else:
            f[iY0,:] = f[iY0,:]-2.*self.bcY[0]/(self.dY[0]*(self.dY[1]+self.dY[0]))
        if self.bcNeumannY[1]:
            f[iY-1,:] = self.bcY[1]*self.dY[-1]
        else:
            f[iY-1,:] = f[iY0,:]-2.*self.bcY[1]/(self.dY[-1]*(self.dY[-1]+self.dY[-2]))

        F = f[iY0:iY,iX0:iX]

        return F

    def forcing3D(self, f):

        # lbcX/Y/Z are used to create slices of f
        # if not Neumann (ie Dirichlet) then omit the endpoint

        # X Boundary Conditions
        # The lower x boundary is found at all y,z and the first x index: f[:,:,iX0]
        # The upper x boundary is found at all y,z and the last x index: f[:,:,iX]
        iX0 = 1 if not self.bcNeumannX[0] else 0
        iX = self.nX-1 if not self.bcNeumannX[1] else self.nX

        if self.bcNeumannX[0]:
            f[:,:,iX0] = self.bcX[0]*self.dX[0]
        else:
            f[:,:,iX0] = f[:,:,iX0]-2.*self.bcX[0]/(self.dX[0]*(self.dX[1]+self.dX[0]))
        if self.bcNeumannX[1]:
            f[:,:,iX-1] = self.bcX[1]*self.dX[-1]
        else:
            f[:,:,iX-1] = f[:,:,iX-1]-2.*self.bcX[1]/(self.dX[-1]*(self.dX[-1]+self.dX[-2]))

        # Y Boundary Conditions
        # The lower y boundary is found at all x,z and the first y index: f[:,iY0,:]
        # The upper y boundary is found at all x,z and the last y index: f[:,iY,:]
        iY0 = 1 if not self.bcNeumannY[0] else 0
        iY = self.nY-1 if not self.bcNeumannY[1] else self.nY

        if self.bcNeumannY[0]:
            f[:,iY0,:] = self.bcY[0]*self.dY[0]
        else:
            f[:,iY0,:] = f[:,iY0,:]-2.*self.bcY[0]/(self.dY[0]*(self.dY[1]+self.dY[0]))
        if self.bcNeumannY[1]:
            f[:,iY-1,:] = self.bcY[1]*self.dY[-1]
        else:
            f[:,iY-1,:] = f[:,iY0,:]-2.*self.bcY[1]/(self.dY[-1]*(self.dY[-1]+self.dY[-2]))

        # Z Boundary Conditions
        # The lower z boundary is found at all x,y and the first z index: f[iZ0,:,:]
        # The upper z boundary is found at all x,y and the last z index: f[iZ,:,:]
        iZ0 = 1 if not self.bcNeumannZ[0] else 0
        iZ = self.nZ-1 if not self.bcNeumannZ[1] else self.nZ

        if self.bcNeumannZ[0]:
            f[iZ0,:,:] = self.bcZ[0]*self.dZ[0]
        else:
            f[iZ0,:,:] = f[iZ0,:,:]-2.*self.bcZ[0]/(self.dZ[0]*(self.dZ[1]+self.dZ[0]))
        if self.bcNeumannZ[1]:
            f[iZ-1,:,:] = self.bcZ[1]*self.dZ[-1]
        else:
            f[iZ-1,:,:] = f[iZ-1,:,:]-2.*self.bcZ[1]/(self.dZ[-1]*(self.dZ[-1]+self.dZ[-2]))

        F = f[iZ0:iZ,iY0:iY,iX0:iX]
        return F


    def jacobify(self, A):
        L = A.get_shape()
        L = L[0]
        I = identity(L)
        D = A.diagonal()
        Dinv = diags(D**(-1), format="csr")
        preT = -Dinv*A
        T = I+preT
        return preT, T

    def powerMethod(self, A, x=None, N=500):
        """
        Power method on A with a random initial eigenvector x for N interations,
        defaulted to 500
        """
        if x is None:
            x = np.random.rand(A.shape[0])

        y = np.zeros_like(x)
        lamb = np.zeros(N)

        for i in range(0,N):
            y = A.dot(x)
            x = y/np.linalg.norm(y)
            tmp = A.dot(x)
            lamb[i] = np.dot(x,tmp)

        return lamb

    def gelfandsFormula(self,A,N=200):
        """
        Apply Gelfand's Forumula to approximate the spectral radius using the p=1
        matrix norm (see p-norm for matrices)

        p(A) ~ norm(A^N)^(1/N)

        Try N=5000 iterations; if p(A) > 1 after N iterations, double the amount
        of iterations.
        """

        size = A.shape[0]
        B = A.power(2)

        i = 2
        its = N
        p = 1
        while i < its:
            if i**2 < its:
                B = B.power(2)
                i = i**2
                print(i)
            else:
                B = B.dot(A)
                i = i+1
            p = norm(B,ord=1)**(1./i)
            if i%1000 == 0:
                print(i)
            if i == its and p >= 1:
                its = 2*its
                print(its)

        return p

    def jacobisMethod(self, decPoints, phi0=None):
        """
        Apply Jacobi's Iterative Method until the initial error has been reduced
        by decPoints number of decimal points

        phi0 is Cartesian indexed, ie phi0[Y,X,Z]

        """
        # Flatten the forcing function into shape of model matrix
        # F = self.f.flatten(order="F")
        F = self.f.flatten()

        # Forcing function for Jacobi's Method is D**(-1) f
        print(self.A.diagonal().shape,F.shape)
        F = (self.A.diagonal())**(-1)*F

        if phi0 is None:
            phi0 = np.zeros_like(self.f)

        if self.dims > 1:
            phi0 = np.moveaxis(phi0,0,1)
        phi = phi0
        phi = phi.flatten()

        # Num of iterations
        t = -decPoints*np.log(10)/np.log(self.rhoT)
        print("Total num of iterations: {}".format(t))
        self.t = t

        for i in range(0,int(t)):
            phi = self.T*phi+F


        # Reshape into the shape of the domain
        soln = np.reshape(phi, self.f.shape)

        self.soln = soln

        return soln




