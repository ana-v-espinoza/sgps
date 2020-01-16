import numpy as np

class StretchedGridPoisson:
    
    """
    Numerically solves a Poisson equation laplacian(phi) = f
    for some scalar phi and forcing function f. The solution
    is found using the method of relaxation on the interior
    of the domain given by the arrays x, y, and z.

    The arrays x, y, and z must be monotonically increasing
    and may have uniform or irregular spacing between elements.

    The solution is subject to the boundary conditions set by
    the user.
    """

    def __init__(self,x,y,z,phi,f):
        # Default the solver to some Poisson Equation
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.z = np.copy(z)

        self.centersToEdges() # Create edges

        self.phi = np.copy(phi) # Initial guess at the solution
        self.f = np.copy(f) # Forcing function

        # Array sizes nx-1, ny-1, and nz-1
        self.dX = x[1:]-x[:-1]
        self.dY = y[1:]-y[:-1]
        self.dZ = z[1:]-z[:-1]
        
        # Make some definitions for some coefficients used in the
        # calculations
        self.Lx = self.dX[:-1]*self.dX[1:]*(self.dX[:-1]+self.dX[1:])
        self.Ly = self.dY[:-1]*self.dY[1:]*(self.dY[:-1]+self.dY[1:])
        self.Lz = self.dZ[:-1]*self.dZ[1:]*(self.dZ[:-1]+self.dZ[1:])
        
        A = self.dX[1:]/self.Lx; B = -(self.dX[:-1]+self.dX[1:])/self.Lx; C = self.dX[:-1]/self.Lx
        D = self.dY[1:]/self.Ly; E = -(self.dY[:-1]+self.dY[1:])/self.Ly; F = self.dY[:-1]/self.Ly
        G = self.dZ[1:]/self.Lz; H = -(self.dZ[:-1]+self.dZ[1:])/self.Lz; J = self.dZ[:-1]/self.Lz
        
        # Reshape the X, Y, and Z coefficients so that elementwise
        # multiplication is done correctly in the iterative solver.
        # For more information see the python docs on "broadcasting"
        
        # Reshape all the x terms
        self.A = A[:,np.newaxis,np.newaxis];
        self.B = B[:,np.newaxis,np.newaxis];
        self.C = C[:,np.newaxis,np.newaxis];
        
        # Reshape all the y terms
        self.D = D[np.newaxis,:,np.newaxis];
        self.E = E[np.newaxis,:,np.newaxis];
        self.F = F[np.newaxis,:,np.newaxis];
        
        # Reshape all the z terms
        self.G = G[np.newaxis,np.newaxis,:];
        self.H = H[np.newaxis,np.newaxis,:];
        self.J = J[np.newaxis,np.newaxis,:];

        # Used to check whether boundary conditions have been set
        self.boundaryCondSet = False

        self.neumBot, self.neumTop = False, False
        self.neumLef, self.neumRig = False, False
        self.neumFor, self.neumBac = False, False

    def centersToEdges(self):
        """
        Take a 1D array of points and return edges by calculating
        the midpoints between consecutive points.
        
        Left and right edges are calculated using the same distance
        from the first (last) center to the first (last) edge
        """
        
        xEdge = (self.x[1:]+self.x[:-1])/2.
        leftEdge = 2*self.x[0]-xEdge[0]
        rightEdge = 2*self.x[-1]-xEdge[-1]
        xEdge = np.concatenate(([leftEdge],xEdge,[rightEdge]))

        yEdge = (self.y[1:]+self.y[:-1])/2.
        leftEdge = 2*self.y[0]-yEdge[0]
        rightEdge = 2*self.y[-1]-yEdge[-1]
        yEdge = np.concatenate(([leftEdge],yEdge,[rightEdge]))

        zEdge = (self.z[1:]+self.z[:-1])/2.
        leftEdge = 2*self.z[0]-zEdge[0]
        rightEdge = 2*self.z[-1]-zEdge[-1]
        zEdge = np.concatenate(([leftEdge],zEdge,[rightEdge]))
        
        self.xEdge, self.yEdge, self.zEdge = xEdge, yEdge, zEdge

        return xEdge,yEdge,zEdge

    def setInitialGuess(self, phi):
        """
        Set the initial guess of your solution, phi. This is done in
        the class constructor, but can be set to a different array
        here.

        phi must have shape (nx, ny, nz), where nx, ny, and nz are the
        sizes of the x, y, and z arrays that make up the domain
        """

        # I could write a check to ensure the shape is correct, but
        # I can't be bothered at this point.
        # I also don't know whether I need to do phi.copy(), or if
        # self.phi = phi is proper syntax to make sure the values are
        # copied to self.phi correctly, but I do it with the
        # self.dX, self.dY, and self.dZ variables in the __init__()
        # function. If something breaks anywhere, check on the syntax
        # for this
        self.phi = phi

    def setForcing(self, f):
        """
        Set the value of the forcing function. This is done in the
        class constuctor, but can be set to a different array here.

        f must have shape (nx, ny, nz), where nx, ny, and nz are the
        sizes of the x, y, and z arrays that make up the domain
        """

        # I could write a check to ensure the shape is correct, but
        # I can't be bothered at this point.
        self.f = f

    def setDirichlet(self,values,boundary):
        """
        Set Dirichlet boundary conditions at the boundary
        given by the "boundary" argument.

        boundary -- one of the following strings (defaults to "back", even when
        the string is not one of the strings found below):

        "bottom": the xy plane at z[0]
        "top": the xy plane at z = z[-1]
        "left": the yz plane at x[0]
        "right": the yz plane at x[-1]
        "forward": the xz plane at y[0]
        "back": the xz plane at y[-1]
        "all": all of the above boundaries
        """

        self.boundaryCondSet = True

        # Eventually write a check to ensure the size of the boundary and that
        # of the values you're trying to set are equal
        if boundary == "bottom" or boundary == "all":
            self.phi[:,:,0] = values
        elif boundary == "top" or boundary == "all":
            self.phi[:,:,-1] = values
        elif boundary == "left" or boundary == "all":
            self.phi[0,:,:] = values
        elif boundary == "right" or boundary == "all":
            self.phi[-1,:,:] = values
        elif boundary == "front" or boundary == "all":
            self.phi[:,0,:] = values
        elif boundary == "back" or boundary == "all":
            self.phi[:,-1,:] = values
        else:
            # Print "error" message, because I never learned proper error handling
            print("ERROR: the string \"%s\" is not one of the allowed options:
                    bottom, top, left, right, front, back, all" % boundary)

    def resetDirichlet(self,boundary):
        """
        Reset Dirichlet boundary conditions at the boundary
        given by the "boundary" argument to make them homogenous

        boundary -- one of the following strings:

        "bottom": the xy plane at z[0]
        "top": the xy plane at z = z[-1]
        "left": the yz plane at x[0]
        "right": the yz plane at x[-1]
        "forward": the xz plane at y[0]
        "back": the xz plane at y[-1]
        "all": all of the above boundaries
        """

        # Eventually write a check to ensure the size of the boundary and that
        # of the values you're trying to set are equal
        if boundary == "bottom" or boundary == "all":
            self.phi[:,:,0] = np.zeros_like(self.phi[:,:,0])
        elif boundary == "top" or boundary == "all":
            self.phi[:,:,-1] = np.zeros_like(self.phi[:,:,-1])
        elif boundary == "left" or boundary == "all":
            self.phi[0,:,:] = np.zeros_like(self.phi[0,:,:])
        elif boundary == "right" or boundary == "all":
            self.phi[-1,:,:] = np.zeros_like(self.phi[-1,:,:])
        elif boundary == "front" or boundary == "all":
            self.phi[:,0,:] = np.zeros_like(self.phi[:,0,:])
        elif boundary == "back" or boundary == "all":
            self.phi[:,-1,:] = np.zeros_like(self.phi[:,-1,:])
        else:
            # Print "error" message, because I never learned proper error handling
            print("ERROR: the string \"%s\" is not one of the allowed options:
                    bottom, top, left, right, front, back, all" % boundary)

    def setNeumann(self,H,boundary):
        """
        Set Neumman boundary conditions dphi/dxi = h(x,y,z,phi,f)
        where dphi/dxi is the partial derivative in the normal
        direction of the boundary and H is a user defined function

        boundary -- one of the following strings:

        "bottom": the xy plane at z[0]
        "top": the xy plane at z = z[-1]
        "left": the yz plane at x[0]
        "right": the yz plane at x[-1]
        "forward": the xz plane at y[0]
        "back": the xz plane at y[-1]
        "all": all of the above boundaries
        """

        if boundary == "bottom" or boundary == "all":
            self.neumBot = True
            self.neumBotH = H
        elif boundary == "top" or boundary == "all":
            self.neumTop = True
            self.neumTopH = H
        elif boundary == "left" or boundary == "all":
            self.neumLef = True
            self.neumLefH = H
        elif boundary == "right" or boundary == "all":
            self.neumRig = True
            self.neumRigH = H
        elif boundary == "forward" or boundary == "all":
            self.neumFor = True
            self.neumForH = H
        elif boundary == "back" or boundary == "all":
            self.neumBac = True
            self.neumBacH = H
        else:
            # Print "error" message, because I never learned proper error handling
            print("ERROR: the string \"%s\" is not one of the allowed options:
                    bottom, top, left, right, front, back, all" % boundary)

        self.boundaryCondSet = True

    def resetNeumann(self,boundary):
        """
        Reset Neumman boundary conditions so that the solver does not
        attempt to use them

        boundary -- one of the following strings:

        "bottom": the xy plane at z[0]
        "top": the xy plane at z = z[-1]
        "left": the yz plane at x[0]
        "right": the yz plane at x[-1]
        "forward": the xz plane at y[0]
        "back": the xz plane at y[-1]
        "all": all of the above boundaries
        """

        H = lambda x,y,z,phi,f : 0

        if boundary == "bottom" or boundary == "all":
            self.neumBot = False
        elif boundary == "top" or boundary == "all":
            self.neumTop = False
        elif boundary == "left" or boundary == "all":
            self.neumLef = False
        elif boundary == "right" or boundary == "all":
            self.neumRig = False
        elif boundary == "forward" or boundary == "all":
            self.neumFor = False
        elif boundary == "back" or boundary == "all":
            self.neumBac = False
        else:
            # Print "error" message, because I never learned proper error handling
            print("ERROR: the string \"%s\" is not one of the allowed options:
                    bottom, top, left, right, front, back, all" % boundary)

    def solvePoisson(self, numOfIt):

        if not self.boundaryCondSet:
            print("Boundary conditions not explicitly set. Defaulting to homogeneous Dirichlet conditions at all boundaries...")

        # Make a copy of self.phi to ensure you can change
        # boundary conditions, numOfIt, etc during the same run
        # without having to create another instance of this object
        self.soln = self.phi.copy()

        # Begin iterative solver
        for i in range(0,numOfIt):
        
            if i % 200 == 0:
                print("Iteration: ", i)
        
            # Calculate boundary conditions (if they are Neumman) using
            # a first order accurate foward or backward difference method
            if self.neumBot:
                self.soln[:,:,0] = self.soln[:,:,0]-self.dZ[0]*self.neumBotH(self.x,self.y,self.z[0],self.soln[:,:,0],self.f[:,:,0])
            if self.neumTop:
                self.soln[:,:,-1] = self.soln[:,:,-2]+self.dZ[-1]*self.neumTopH(self.x,self.y,self.z[-1],self.soln[:,:,-1],self.f[:,:,-1])
            if self.neumLef:
                self.soln[0,:,:] = self.soln[1,:,:]-self.dX[0]*self.neumLefH(self.x[0],self.y,self.z,self.soln[0,:,:],self.f[0,:,:])
            if self.neumRig:
                self.soln[-1,:,:] = self.soln[-2,:,:]+self.dX[-1]*self.neumRigH(self.x[-1],self.y,self.z,self.soln[-1,:,:],self.f[-1,:,:])
            if self.neumFor:
                self.soln[:,0,:] = self.soln[:,1,:]-self.dY[0]*self.neumForH(self.x,self.y[0],self.z,self.soln[:,0,:],self.f[:,0,:])
            if self.neumBac:
                self.soln[:,-1,:] = self.soln[:,-2,:]+self.dY[0]*self.neumBacH(self.x,self.y[-1],self.z,self.soln[:,-1,:],self.f[:,-1,:])
        
            # Apply the method of relaxation for interior points
            self.soln[1:-1,1:-1,1:-1] = 1./(self.B+self.E+self.H)*(
                    self.f[1:-1,1:-1,1:-1]
                    - self.A*self.soln[:-2,1:-1,1:-1] - self.C*self.soln[2:,1:-1,1:-1]
                    - self.D*self.soln[1:-1,:-2,1:-1] - self.F*self.soln[1:-1,2:,1:-1]
                    - self.G*self.soln[1:-1,1:-1,:-2] - self.J*self.soln[1:-1,1:-1,2:])

    def saveSolution(self, filename):
        """
        Save the following variables in the .npz file format:
        x,y,z,
        xEdge,yEdge,zEdge,
        phi,soln
        """
        np.savez_compressed(filename, x=self.x, y=self.y, z=self.z,
                xEdge=self.xEdge, yEdge=self.yEdge, zEdge=self.zEdge,
                phiInit=self.phi, soln=self.soln)
