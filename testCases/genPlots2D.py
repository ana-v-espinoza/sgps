import numpy as np
from sgpsKron import *
from matplotlib import pyplot as plt

def centersToEdges(X):
    """
    Take a 1D array of points and return edges by calculating
    the midpoints between consecutive points.
    
    Left and right edges are calculated using the same distance
    from the first (last) center to the first (last) edge
    """
    
    xEdge = (X[1:]+X[:-1])/2.
    leftEdge = 2*X[0]-xEdge[0]
    rightEdge = 2*X[-1]-xEdge[-1]
    xEdge = np.concatenate(([leftEdge],xEdge,[rightEdge]))

    return xEdge

def genPlots(sgps,exactSoln,bcType,bcValues,decPlaces,const,gridType):

    nX = sgps.X.size
    nY = sgps.Y.size

    X = sgps.X
    Y = sgps.Y

    mX, mY = np.meshgrid(X,Y)
    eX = centersToEdges(mX)
    eY = centersToEdges(mY)

    if bcType == "dirichlet":
        Ax = sgps.set_1DA("X1")
        Ay = sgps.set_1DA("X2")

        # Do not explicitly set X2 boundary values, they default to homogenous
        # Dirichlet
        sgps.set_boundaryValues("X1","lower", bcValues[0])
        sgps.set_boundaryValues("X1","upper", bcValues[1])

        A, T, rhoT = sgps.set_modelMatrix()
        f = np.zeros((nY,nX))
        sgps.set_forcing(f)
        phi0 = const*np.ones((nX-2,nY-2))
        soln = sgps.jacobisMethod(decPlaces,phi0=phi0)

        exact = exactSoln(mX,mY,bcValues[0], bcValues[1])[1:-1,1:-1]


        err0 = exact-phi0
        nErr0 = np.linalg.norm(err0)
        err = exact - soln
        nErr = np.linalg.norm(err)

        print("Spectral Radius: {}".format(sgps.rhoT))
        print("err0, err, err/err0: {} {} {}".format(nErr0, nErr,
            nErr/nErr0))
        print("rhoT^t: {}".format(sgps.rhoT**sgps.t))
        print()

        im = plt.pcolormesh(eX[1:-1,1:-1],eY[1:-1,1:-1],err)

        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Dirichlet Boundary Conditions -- Error (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates2D/{}_{}_error.png".format(bcType,gridType))
        #plt.show()
        plt.close()

        im = plt.pcolormesh(eX[1:-1,1:-1],eY[1:-1,1:-1],soln)
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Dirichlet Boundary Conditions -- Solution (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates2D/{}_{}_soln.png".format(bcType,gridType))
        #plt.show()
        plt.close()

        im = plt.pcolormesh(eX[1:-1,1:-1],eY[1:-1,1:-1],exact)
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Dirichlet Boundary Conditions -- Exact Soln (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates2D/{}_{}_exact.png".format(bcType,gridType))
        #plt.show()
        plt.close()

    if bcType == "LNBC":
        # Lower Neumann BC
        Ax = sgps.set_1DA("X1", lbcNeumann=True)
        Ay = sgps.set_1DA("X2")
        sgps.set_boundaryValues("X1","lower", bcValues[0])
        sgps.set_boundaryValues("X1","upper", bcValues[1])
        sgps.set_modelMatrix()
        f = np.zeros((nY,nX))
        sgps.set_forcing(f)
        phi0 = const*np.ones((nY-2,nX-1))
        soln = sgps.jacobisMethod(decPlaces,phi0=phi0)

        exact = exactSoln(mX,mY,bcValues[0], bcValues[1])[1:-1,:-1]
        err0 = exact - phi0
        nErr0 = np.linalg.norm(err0)
        err = exact - soln
        nErr = np.linalg.norm(err)

        print("Spectral Radius: {}".format(sgps.rhoT))
        print("err0, err, err/err0: {} {} {}".format(nErr0, nErr,
            nErr/nErr0))
        print("rhoT^t: {}".format(sgps.rhoT**sgps.t))
        print()

        im = plt.pcolormesh(eX[1:-1,:-1],eY[1:-1,:-1],err)
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("West Neumann Boundary Conditions -- Error (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates2D/{}_{}_error.png".format(bcType,gridType))
        plt.colorbar(im)
        #plt.show()
        plt.close()

        im = plt.pcolormesh(eX[1:-1,:-1],eY[1:-1,:-1],soln)
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("West Neumann Boundary Conditions -- Solution (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates2D/{}_{}_soln.png".format(bcType,gridType))
        #plt.show()
        plt.close()

        im = plt.pcolormesh(eX[1:-1,:-1],eY[1:-1,:-1],exact)
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("West Neumann Boundary Conditions -- Exact Soln (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates2D/{}_{}_exact.png".format(bcType,gridType))
        #plt.show()
        plt.close()

    if bcType == "UNBC":
        # Upper Neumann BC
        Ax = sgps.set_1DA("X1", ubcNeumann=True)
        Ay = sgps.set_1DA("X2")
        sgps.set_boundaryValues("X1","lower", bcValues[0])
        sgps.set_boundaryValues("X1","upper", bcValues[1])
        sgps.set_modelMatrix()
        f = np.zeros((nY,nX))
        sgps.set_forcing(f)
        phi0 = const*np.ones((nY-2, nX-1))
        soln = sgps.jacobisMethod(decPlaces,phi0=phi0)

        exact = exactSoln(mX,mY,bcValues[0], bcValues[1])[1:-1,1:]
        err0 = exact - phi0
        nErr0 = np.linalg.norm(err0)
        err = exact - soln
        nErr = np.linalg.norm(err)
        print("Spectral Radius: {}".format(sgps.rhoT))
        print("err0, err, err/err0: {} {} {}".format(nErr0, nErr,
            nErr/nErr0))
        print("rhoT^t: {}".format(sgps.rhoT**sgps.t))
        print()

        im = plt.pcolormesh(eX[1:-1,1:],eY[1:-1,1:],err)
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("East Neumann Boundary Conditions -- Error (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates2D/{}_{}_error.png".format(bcType,gridType))
        #plt.show()
        plt.close()

        im = plt.pcolormesh(eX[1:-1,1:],eY[1:-1,1:],soln)
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("East Neumann Boundary Conditions -- Solution (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates2D/{}_{}_soln.png".format(bcType,gridType))
        #plt.show()
        plt.close()

        im = plt.pcolormesh(eX[1:-1,1:],eY[1:-1,1:],exact)
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("East Neumann Boundary Conditions -- Exact Soln (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates2D/{}_{}_exact.png".format(bcType,gridType))
        #plt.show()
        plt.close()
