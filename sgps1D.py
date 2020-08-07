import numpy as np

class StretchedGridPoisson1D:
    
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

    def __init__(self,x,y,phi,f):
        # Default the solver to some Poisson Equation
        self.x, self.nx = np.copy(x), x.size

        self.centersToEdges() # Create edges

        self.phi = np.copy(phi) # Initial guess at the solution
        self.f = np.copy(f) # Forcing function

        # Array sizes nx-1, ny-1, and nz-1
        self.dX = x[1:]-x[:-1]
        
        # Make some definitions for some coefficients used in the
        # calculations
        self.Lx = self.dX[:-1]*self.dX[1:]*(self.dX[:-1]+self.dX[1:])
        
        A = 2*self.dX[1:]/self.Lx; B = -2*(self.dX[:-1]+self.dX[1:])/self.Lx; C = 2*self.dX[:-1]/self.Lx
        
        # Reshape the X coefficients so that elementwise
        # multiplication is done correctly in the iterative solver.
        # For more information see the python docs on "broadcasting"
        
        # Reshape all the x terms
        self.A = A[:,np.newaxis,np.newaxis];
        self.B = B[:,np.newaxis,np.newaxis];
        self.C = C[:,np.newaxis,np.newaxis];
        
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

        self.xEdge = xEdge

        return xEdge

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

        boundary -- one of the following strings

        "left": the line at x[0]
        "right": the line at x[-1]
        "all": all of the above boundaries
        """

        self.boundaryCondSet = True

        # Eventually write a check to ensure the size of the boundary and that
        # of the values you're trying to set are equal
        if boundary == "left" or boundary == "all":
            self.phi[0] = values
        elif boundary == "right" or boundary == "all":
            self.phi[-1] = values
        else:
            # Print "error" message, because I never learned proper error handling
            print("ERROR: the string \"%s\" is not one of the allowed options:\
                    left, right, all" % boundary)

    def resetDirichlet(self,boundary):
        """
        Reset Dirichlet boundary conditions at the boundary
        given by the "boundary" argument to make them homogenous

        boundary -- one of the following strings

        "left": the line at x[0]
        "right": the line at x[-1]
        "all": all of the above boundaries
        """

        # Eventually write a check to ensure the size of the boundary and that
        # of the values you're trying to set are equal
        if boundary == "left" or boundary == "all":
            self.phi[0] = np.zeros_like(self.phi[0])
        elif boundary == "right" or boundary == "all":
            self.phi[-1] = np.zeros_like(self.phi[-1])
        else:
            # Print "error" message, because I never learned proper error handling
            print("ERROR: the string \"%s\" is not one of the allowed options:\
                    left, right, all" % boundary)

    def setNeumann(self,H,boundary):
        """
        Set Neumman boundary conditions dphi/dxi = h(x,y,z,phi,f)
        where dphi/dxi is the partial derivative in the normal
        direction of the boundary and H is a user defined function

        boundary -- one of the following strings

        "left": the line at x[0]
        "right": the line at x[-1]
        "all": all of the above boundaries
        """

        if boundary == "left" or boundary == "all":
            self.neumLef = True
            self.neumLefH = H
        elif boundary == "right" or boundary == "all":
            self.neumRig = True
            self.neumRigH = H
        else:
            # Print "error" message, because I never learned proper error handling
            print("ERROR: the string \"%s\" is not one of the allowed options:\
                    left, right all" % boundary)

        self.boundaryCondSet = True

    def resetNeumann(self,boundary):
        """
        Reset Neumman boundary conditions so that the solver does not
        attempt to use them

        boundary -- one of the following strings

        "left": the line at x[0]
        "right": the line at x[-1]
        "all": all of the above boundaries
        """

        H = lambda x,phi,f : 0

        if boundary == "left" or boundary == "all":
            self.neumLef = False
            self.neumLefH = H
        elif boundary == "right" or boundary == "all":
            self.neumRig = False
            self.neumRigH = H
        else:
            # Print "error" message, because I never learned proper error handling
            print("ERROR: the string \"%s\" is not one of the allowed options:\
                    left, right, all" % boundary)

    def solvePoisson(self, numOfIt, debug=False, dbFilename="./debug.npz"):

        if not self.boundaryCondSet:
            print("Boundary conditions not explicitly set. Defaulting to homogeneous Dirichlet conditions at all boundaries...")

        # Make a copy of self.phi to ensure you can change
        # boundary conditions, numOfIt, etc during the same run
        # without having to create another instance of this object
        self.soln = self.phi.copy()

        # Begin iterative solver
        print("Iteration: ", end="")
        for i in range(0,numOfIt):
        
            if i % 200 == 0:
                print(i, end=", ")
        
            # Calculate boundary conditions (if they are Neumman) using
            # a first order accurate foward or backward difference method
            if self.neumLef:
                self.soln[0] = self.soln[1]-self.dX[0]*self.neumLefH(self.x[0],self.soln[0],self.f[0])
            if self.neumRig:
                self.soln[-1] = self.soln[-2]-self.dX[-1]*self.neumLefH(self.x[-1],self.soln[-1],self.f[-1])
        
            # Apply the method of relaxation for interior points
            self.soln[1:-1,1:-1] = 1./(self.B)*(
                    self.f[1:-1]
                    - self.A*self.soln[:-2,1:-1] - self.C*self.soln[2:,1:-1])
        print()

    def saveSolution(self, filename):
        """
        Save the following variables in the .npz file format:
        x,
        xEdge,
        phi,soln
        """
        np.savez_compressed(filename, x=self.x,
                xEdge=self.xEdge,
                phiInit=self.phi, soln=self.soln)
