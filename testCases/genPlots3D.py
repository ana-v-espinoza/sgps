import numpy as np
from sgps import *
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
    nZ = sgps.Z.size

    X = sgps.X
    Y = sgps.Y
    Z = sgps.Z

    #mZ, mY, mX = np.meshgrid(Z,Y,X)
    mX, mY, mZ = np.meshgrid(X,Y,Z)
    eX = centersToEdges(mX)
    eY = centersToEdges(mY)
    eZ = centersToEdges(mZ)

    if bcType == "dirichlet":
        Ax = sgps.set_1DA("X1")
        Ay = sgps.set_1DA("X2")
        Az = sgps.set_1DA("X3")

        # Do not explicitly set X2 boundary values, they default to homogenous
        # Dirichlet
        sgps.set_boundaryValues("X1","lower", bcValues[0])
        sgps.set_boundaryValues("X1","upper", bcValues[1])
        f = np.zeros((nZ,nY,nX))
        sgps.set_forcing(f)

        A, T, rhoT = sgps.set_modelMatrix()

        phi0 = const*np.ones((nZ-2,nY-2,nX-2))
        soln = sgps.jacobisMethod(decPlaces,phi0=phi0)
        soln = np.moveaxis(soln,0,-1)

        phi0 = np.moveaxis(phi0,0,-1)
        exact = exactSoln(mX,mY,mZ,bcValues[0], bcValues[1])[1:-1,1:-1,1:-1]

        err0 = exact-phi0
        nErr0 = np.linalg.norm(err0)
        err = exact - soln
        nErr = np.linalg.norm(err)

        print("Spectral Radius: {}".format(sgps.rhoT))
        print("err0, err, err/err0: {} {} {}".format(nErr0, nErr,
            nErr/nErr0))
        print("rhoT^t: {}".format(sgps.rhoT**sgps.t))
        print()

        print("err")
        im = plt.pcolormesh(
                eX[1:-1,1:-1,0],eY[1:-1,1:-1,0],err[:,:,0])
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Dirichlet Boundary Conditions -- Error (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates3D/{}_{}_error.png".format(bcType,gridType))
        #plt.show()
        plt.close()

        print("soln")
        im = plt.pcolormesh(eX[1:-1,1:-1,0],eY[1:-1,1:-1,0],soln[:,:,int(nZ/2)])
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Dirichlet Boundary Conditions -- Solution (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates3D/{}_{}_soln.png".format(bcType,gridType))
        #plt.show()
        plt.close()

        print("exact")
        im = plt.pcolormesh(eX[1:-1,1:-1,0],eY[1:-1,1:-1,0],exact[:,:,int(nZ/2)])
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Dirichlet Boundary Conditions -- Exact Soln (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates3D/{}_{}_exact.png".format(bcType,gridType))
        #plt.show()
        plt.close()

    if bcType == "LNBC":
        # Lower Neumann BC
        Ax = sgps.set_1DA("X1", lbcNeumann=True)
        Ay = sgps.set_1DA("X2")
        Az = sgps.set_1DA("X3")

        sgps.set_boundaryValues("X1","lower", bcValues[0])
        sgps.set_boundaryValues("X1","upper", bcValues[1])
        f = np.zeros((nZ,nY,nX))
        F = sgps.set_forcing(f)

        sgps.set_modelMatrix()

        # Solution indexed as soln[z,y,x]
        phi0 = const*np.ones((nZ-2,nY-2,nX-1))
        soln = sgps.jacobisMethod(decPlaces,phi0=phi0)
        soln = np.moveaxis(soln,0,-1)

        phi0 = np.moveaxis(phi0,0,-1)
        exact = exactSoln(mX,mY,mZ,bcValues[0], bcValues[1])[1:-1,:-1,1:-1]

        err0 = exact - phi0
        nErr0 = np.linalg.norm(err0)
        err = exact - soln
        nErr = np.linalg.norm(err)

        print("Spectral Radius: {}".format(sgps.rhoT))
        print("err0, err, err/err0: {} {} {}".format(nErr0, nErr,
            nErr/nErr0))
        print("rhoT^t: {}".format(sgps.rhoT**sgps.t))
        print()

        print("err")
        im = plt.pcolormesh(
                eX[1:-1,:-1,0],eY[1:-1,:-1,0],err[:,:,int(nZ/2)])
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("West Neumann Boundary Conditions -- Error (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates3D/{}_{}_error.png".format(bcType,gridType))
        #plt.show()
        plt.close()

        print("soln")
        im = plt.pcolormesh(eX[1:-1,:-1,0],eY[1:-1,:-1,0],soln[:,:,int(nZ/2)])
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("West Neumann Boundary Conditions -- Solution (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates3D/{}_{}_soln.png".format(bcType,gridType))
        #plt.show()
        plt.close()

        print("exact")
        im = plt.pcolormesh(eX[1:-1,:-1,0],eY[1:-1,:-1,0],exact[:,:,int(nZ/2)])
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("West Neumann Boundary Conditions -- Exact Soln (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates3D/{}_{}_exact.png".format(bcType,gridType))
        #plt.show()
        plt.close()

    if bcType == "UNBC":
        # Upper Neumann BC
        Ax = sgps.set_1DA("X1", ubcNeumann=True)
        Ay = sgps.set_1DA("X2")
        Az = sgps.set_1DA("X3")
        sgps.set_boundaryValues("X1","lower", bcValues[0])
        sgps.set_boundaryValues("X1","upper", bcValues[1])
        sgps.set_modelMatrix()
        f = np.zeros((nZ,nY,nX))
        sgps.set_forcing(f)
        phi0 = const*np.ones((nZ-2,nY-2,nX-1))
        soln = sgps.jacobisMethod(decPlaces,phi0=phi0)
        # Reshape to Cartesian indexing [Y,X,Z] (for some reason)
        soln = np.moveaxis(soln,0,-1)

        phi0 = np.moveaxis(phi0,0,-1)
        exact = exactSoln(mX,mY,mZ,bcValues[0], bcValues[1])[1:-1,1:,1:-1]
        err0 = exact - phi0
        nErr0 = np.linalg.norm(err0)
        err = exact - soln
        nErr = np.linalg.norm(err)
        print("Spectral Radius: {}".format(sgps.rhoT))
        print("err0, err, err/err0: {} {} {}".format(nErr0, nErr,
            nErr/nErr0))
        print("rhoT^t: {}".format(sgps.rhoT**sgps.t))
        print()

        print("err")
        im = plt.pcolormesh(
                eX[1:-1,1:,0],eY[1:-1,1:,0],err[:,:,int(nZ/2)])
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("East Neumann Boundary Conditions -- Error (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates3D/{}_{}_error.png".format(bcType,gridType))
        #plt.show()
        plt.close()

        print("soln")
        im = plt.pcolormesh(eX[1:-1,1:,0],eY[1:-1,1:,0],soln[:,:,int(nZ/2)])
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("East Neumann Boundary Conditions -- Solution (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates3D/{}_{}_soln.png".format(bcType,gridType))
        #plt.show()
        plt.close()

        print("exact")
        im = plt.pcolormesh(eX[1:-1,1:,0],eY[1:-1,1:,0],exact[:,:,int(nZ/2)])
        plt.colorbar(im)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("East Neumann Boundary Conditions -- Exact Soln (shaded)")
        plt.tight_layout(rect=[0,0.05,0.95,0.95])
        plt.savefig("parallelPlates3D/{}_{}_exact.png".format(bcType,gridType))
        #plt.show()
        plt.close()
