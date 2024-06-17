from parareal.parareal.mpi_parareal import parareal
from parareal.lorenz.lorenz import lorenz
from parareal.solver.rk4 import rk4
from parareal.solver.rk2 import rk2
from parareal.solver.feuler import feuler
from parareal.solver.beuler import beuler
from parareal.plot.plot3d import plot3d
from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt


# Model parameters
sigma, rho, beta = 10.0, 28.0, 8/3
U0 = np.array([5.0, -5.0, 20.0])
tspan = [0, 1]
lorenz_ = lambda t,state:lorenz(t, state, sigma, rho, beta)

#Parareal params
N = 10
tol = 1e-10
max_iter = 10
G_Nh=1
F_Nh=80
G = lambda tspan,u0, :rk4(lorenz_, tspan, u0, G_Nh)[1][-1]
F = lambda tspan,u0, :rk4(lorenz_, tspan, u0, F_Nh)[1][-1]

# Run Parareal
start = time.time()
t, iterations, solution = parareal(G, F, tspan, U0, N, max_iter, tol)
if MPI.COMM_WORLD.Get_rank() == 0:
    end = time.time()
    print("time = ",end - start)
    print(f'Solution after {iterations} iterations: ')
    print(solution.shape)
    fig, ax = plt.subplots()
    ax.plot(t,solution[:,0])
    ax.plot(t,solution[:,1])
    ax.plot(t,solution[:,2])
    plt.show()
    