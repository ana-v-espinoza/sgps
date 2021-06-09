import numpy as np
from sgpsKron import *
from matplotlib import pyplot as plt

def genPlots(X,sgps,exactSoln,bcType,bcValues,decPlaces,const):

    nX = X.size

    if bcType == "dirichlet":
        Ax = sgps.set_1DA("X1")
        sgps.set_boundaryValues("X1","lower", bcValues[0])
        sgps.set_boundaryValues("X1","upper", bcValues[1])
        A, T, rhoT = sgps.set_modelMatrix()
        f = np.zeros(nX)
        sgps.set_forcing(f)
        phi0 = const*np.ones(nX-2)
        soln = sgps.jacobisMethod(decPlaces,phi0=phi0)

        exact = exactSoln(X,bcValues[0], bcValues[1])[1:-1]

        err0 = exact-phi0
        nErr0 = np.linalg.norm(err0)
        err = exact - soln
        nErr = np.linalg.norm(err)

        print("Spectral Radius: {}".format(sgps.rhoT))
        print("err0, err, err/err0: {} {} {}".format(nErr0, nErr,
            nErr/nErr0))
        print("rhoT^t: {}".format(sgps.rhoT**sgps.t))
        print()

        plt.plot(X[1:-1],err)
        plt.plot(X[1:-1],soln)
        plt.plot(X[1:-1],exact)
        plt.legend(["err", "soln", "exact"])
        plt.show()
        plt.close()

    if bcType == "LNBC":
        # Lower Neumann BC
        Ax = sgps.set_1DA("X1", lbcNeumann=True)
        sgps.set_boundaryValues("X1","lower", bcValues[0])
        sgps.set_boundaryValues("X1","upper", bcValues[1])
        sgps.set_modelMatrix()
        f = np.zeros(nX)
        sgps.set_forcing(f)
        phi0 = const*np.ones(nX-1)
        soln = sgps.jacobisMethod(decPlaces,phi0=phi0)

        exact = exactSoln(X,bcValues[0], bcValues[1])[:-1]
        err0 = exact - phi0
        nErr0 = np.linalg.norm(err0)
        err = exact - soln
        nErr = np.linalg.norm(err)

        print("Spectral Radius: {}".format(sgps.rhoT))
        print("err0, err, err/err0: {} {} {}".format(nErr0, nErr,
            nErr/nErr0))
        print("rhoT^t: {}".format(sgps.rhoT**sgps.t))
        print()

        plt.plot(X[:-1],err)
        plt.plot(X[:-1],soln)
        plt.plot(X[:-1],exact)
        plt.legend(["err", "soln", "exact"])
        plt.show()
        plt.close()

    if bcType == "UNBC":
        # Upper Neumann BC
        Ax = sgps.set_1DA("X1", ubcNeumann=True)
        sgps.set_boundaryValues("X1","lower", bcValues[0])
        sgps.set_boundaryValues("X1","upper", bcValues[1])
        sgps.set_modelMatrix()
        f = np.zeros(nX)
        sgps.set_forcing(f)
        phi0 = const*np.ones(nX-1)
        soln = sgps.jacobisMethod(decPlaces,phi0=phi0)

        exact = exactSoln(X,bcValues[0], bcValues[1])[1:]
        err0 = exact - phi0
        nErr0 = np.linalg.norm(err0)
        err = exact - soln
        nErr = np.linalg.norm(err)
        print("Spectral Radius: {}".format(sgps.rhoT))
        print("err0, err, err/err0: {} {} {}".format(nErr0, nErr,
            nErr/nErr0))
        print("rhoT^t: {}".format(sgps.rhoT**sgps.t))
        print()

        plt.plot(X[1:],err)
        plt.plot(X[1:],soln)
        plt.plot(X[1:],exact)
        plt.legend(["err", "soln", "exact"])
        plt.show()
        plt.close()
        plt.close()
