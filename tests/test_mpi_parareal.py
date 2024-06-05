from parareal.parareal.mpi_parareal import parareal
from parareal.lorenz.lorenz import lorenz
from parareal.solver.feuler import feuler
from mpi4py import MPI
import numpy as np


def test_mpi_parareal():
    # Model parameters
    sigma, rho, beta = 10.0, 28.0, 8/3
    U0 = np.array([1.0, 1.0, 1.0])
    tspan = [0, 60]
    lorenz_ = lambda t,state:lorenz(t, state, sigma, rho, beta)

    #Parareal params
    N = 10000
    tol = 1e-1
    max_iter = 5
    G_Nh=1
    F_Nh=100
    G = lambda tspan,u0, :feuler(lorenz_, tspan, u0, G_Nh)[1][-1]
    F = lambda tspan,u0, :feuler(lorenz_, tspan, u0, F_Nh)[1][-1]

    try:
        parareal(G, F, tspan, U0, N, max_iter, tol)
    except:
        print("Error when executing parareal in Parallel function")