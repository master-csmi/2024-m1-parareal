from parareal.parareal.mpi_parareal import mpi_parareal
from parareal.lorenz.lorenz import lorenz
from parareal.solver.feuler import feuler
from parareal.solver.beuler import beuler
from parareal.solver.rk2 import rk2
from parareal.solver.brk2 import brk2
from parareal.solver.rk4 import rk4
from parareal.solver.brk4 import brk4
from mpi4py import MPI
import numpy as np
import time
import sys


# Model parameters
sigma, rho, beta = 10.0, 28.0, 8/3
U0 = np.array([5, -5, 20])
tspan = [0., 10.]
lorenz_ = lambda t, state: lorenz(t, state, sigma, rho, beta)

# Parareal params
N = 1000
tol = 1e-10
max_iter = 100
G_Nh = 1
F_Nh = 80

# Get solvers from command line arguments
G_solver_name = sys.argv[1]
F_solver_name = sys.argv[2]

# Map solver names to functions
solvers = {
    'explicit_euler': feuler,
    'implicit_euler': beuler,
    'explicit_rk2': rk2,
    'implicit_rk2': brk2,
    'explicit_rk4': rk4,
    'implicit_rk4': brk4,
}

G_solver = solvers[G_solver_name]
F_solver = solvers[F_solver_name]

G = lambda tspan, u0: G_solver(lorenz_, tspan, u0, G_Nh)[1][-1]
F = lambda tspan, u0: F_solver(lorenz_, tspan, u0, F_Nh)[1][-1]

# Run Parareal
t, iterations, solutions, parareal_time = mpi_parareal(G, F, tspan, U0, N, max_iter, tol)

if MPI.COMM_WORLD.Get_rank() == 0 :
    # Print the execution time
    print(f"np={MPI.COMM_WORLD.Get_size()},coarse solver={G_solver_name},fine solver={F_solver_name}, Execution Time: {parareal_time} seconds")

    # Save the time to a file
    with open(f"execution-times-{MPI.COMM_WORLD.Get_size()}-{G_solver_name}-{F_solver_name}.txt", "a") as f:
        f.write(f"{MPI.COMM_WORLD.Get_size()},{G_solver_name},{F_solver_name},{parareal_time}\n")