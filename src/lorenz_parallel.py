from mpi_parareal import *
from lorenz import *
from euler import *
from plot import *
import time
from mpi4py import MPI


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

# Run Parareal
start = time.time()
t, iterations, solution = parareal(G, F, tspan, U0, N, max_iter, tol)
if MPI.COMM_WORLD.Get_rank() == 0:
    end = time.time()
    print("time = ",end - start)
    print(f'Solution after {iterations} iterations: {solution}')
    plot3d(t,solution)