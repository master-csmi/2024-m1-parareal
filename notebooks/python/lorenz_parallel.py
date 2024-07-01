from parareal.parareal.mpi_parareal import mpi_parareal
from parareal.lorenz.lorenz import lorenz
from parareal.solver.feuler import feuler
from parareal.solver.beuler import beuler
from parareal.solver.rk4 import rk4
from parareal.utils.plot3d import plot3d
from mpi4py import MPI
import numpy as np


# Model parameters
sigma, rho, beta = 10.0, 28.0, 8/3
U0 = np.array([5,-5,20])
tspan = [0, 10]
lorenz_ = lambda t,state:lorenz(t, state, sigma, rho, beta)

#Parareal params
N = 180
tol = 1e-10
max_iter = 100
G_Nh=1
F_Nh=80
G = lambda tspan,u0, :rk4(lorenz_, tspan, u0, G_Nh)[1][-1]
F = lambda tspan,u0, :rk4(lorenz_, tspan, u0, F_Nh)[1][-1]

# Run Parareal
t, iterations, solutions, time = mpi_parareal(G, F, tspan, U0, N, max_iter, tol)