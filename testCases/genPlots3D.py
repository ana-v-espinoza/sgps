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

def genPlots(sgps,exactSoln,bcType,bcValues,decPlaces,const):

    nX = sgps.X.size
    nY = sgps.Y.size
    nZ = sgps.Z.size

    X = sgps.X
    Y = sgps.Y
    Z = sgps.Z

    mZ, mY, mX = np.meshgrid(Z,Y,X)
    eX = centersToEdges(mX)
    eY = centersToEdges(mY)
    eZ = centersToEdges(mZ)

    print(eX[0,1:-1,1:-1])
    im = plt.pcolormesh(eX[0,:,:])
    plt.colorbar(im)
    plt.show()
    plt.close()

    return

    if bcType == "dirichlet":
        Ax = sgps.set_1DA("X1")
        Ay = sgps.set_1DA("X2")
        Az = sgps.set_1DA("X3")

        # Do not explicitly set X2 boundary values, they default to homogenous
        # Dirichlet
        sgps.set_boundaryValues("X1","lower", bcValues[0])
        sgps.set_boundaryValues("X1","upper", bcValues[1])

        A, T, rhoT = sgps.set_modelMatrix()
        f = np.zeros((nZ,nY,nX))
        sgps.set_forcing(f)
        phi0 = const*np.ones((nZ-2,nY-2,nX-2))
        soln = sgps.jacobisMethod(decPlaces,phi0=phi0)
        print(soln.shape)
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

        #im = plt.pcolormesh(
        #        eX[int(nZ/2),1:-1,1:-1],
        #        eY[int(nZ/2),1:-1,1:-1],
        #        soln[int(nZ/2),:,:]
        #        )
        im = plt.pcolormesh(soln[int(nZ/2),:,:])
        plt.colorbar(im)
        plt.show()
        plt.close()

        #im = plt.pcolormesh(
        #        eX[1:-1,int(nY/2),1:-1],eY[1:-1,int(nY/2),1:-1],err[:,int(nY/2),:])
        #plt.colorbar(im)
        #plt.show()
        #plt.close()

        #im = plt.pcolormesh(eX[1:-1,int(nY/2),1:-1],eY[1:-1,int(nY/2),1:-1],soln)
        #plt.colorbar(im)
        #plt.show()
        #plt.close()

        #im = plt.pcolormesh(eX[1:-1,int(nY/2),1:-1],eY[1:-1,int(nY/2),1:-1],exact)
        #plt.colorbar(im)
        #plt.show()
        #plt.close()

    if bcType == "LNBC":
        # Lower Neumann BC
        Ax = sgps.set_1DA("X1", lbcNeumann=True)
        Ay = sgps.set_1DA("X2")
        sgps.set_boundaryValues("X1","lower", bcValues[0])
        sgps.set_boundaryValues("X1","upper", bcValues[1])
        sgps.set_modelMatrix()
        f = np.zeros((nX,nY))
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
        plt.show()
        plt.close()

        im = plt.pcolormesh(eX[1:-1,:-1],eY[1:-1,:-1],soln)
        plt.colorbar(im)
        plt.show()
        plt.close()

        im = plt.pcolormesh(eX[1:-1,:-1],eY[1:-1,:-1],exact)
        plt.colorbar(im)
        plt.show()
        plt.close()

    if bcType == "UNBC":
        # Upper Neumann BC
        Ax = sgps.set_1DA("X1", ubcNeumann=True)
        Ay = sgps.set_1DA("X2")
        sgps.set_boundaryValues("X1","lower", bcValues[0])
        sgps.set_boundaryValues("X1","upper", bcValues[1])
        sgps.set_modelMatrix()
        f = np.zeros((nX,nY))
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
        plt.show()
        plt.close()

        im = plt.pcolormesh(eX[1:-1,:-1],eY[1:-1,1:],soln)
        plt.colorbar(im)
        plt.show()
        plt.close()

        im = plt.pcolormesh(eX[1:-1,:-1],eY[1:-1,1:],exact)
        plt.colorbar(im)
        plt.show()
        plt.close()
