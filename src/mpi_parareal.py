import numpy as np
from mpi4py import MPI
from euler import *
from lorenz import *
from plot import *
import time

def parareal(G, F, tspan, y0, N, K, tol=0.5):
    """
    Parareal algorithm to solve a system of differential equations.

    :param G: Coarse solver function
    :param F: Fine solver function
    :param tspan: Array of time (start time, end time)
    :param y0: Initial condition
    :param N: Number of sub-intervals
    :param K: Number of Parareal iterations
    :param tol: Tolerance for stopping criterion
    :return: Approximate solution y at discretization points
    """
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    times = np.linspace(tspan[0], tspan[1], N + 1)
    
    u = [y0]
    for i in range(N):
        u.append(G([times[i], times[i + 1]], u[i]))
    u = np.array(u)
    
    G_ = u.copy()
    F_ = np.zeros_like(u)
    F_[0] = y0
    
    for iter in range(K):
        u_prev = u.copy()
        G_prev = G_.copy()
        
        # Scatter the time slices among the processes
        local_indices = np.array_split(range(N), size)[rank]
        local_F_ = np.zeros_like(F_)
        
        for i in local_indices:
            local_F_[i+1] = F([times[i], times[i + 1]], u[i])
        
        # Gather the fine solver results from all processes
        comm.Allreduce(local_F_, F_, op=MPI.SUM)
        F_[0] = y0
        
        for i in range(N):
            G_[i+1] = G([times[i], times[i + 1]], u[i])
            u[i+1] = G_[i+1] + F_[i+1] - G_prev[i+1]
        
        # Check the stopping criterion
        error = np.linalg.norm(u - u_prev)/len(times)
        if rank == 0:
            print(f"Iteration {iter}, Error: {error}")
        if error < tol:
            break
        
    
    return times, iter, u

# Example usage
# Model parameters
sigma, rho, beta = 10.0, 28.0, 8/3
lorenz_ = lambda t,state:lorenz(t, state, sigma, rho, beta)
U0 = np.array([1.0, 1.0, 1.0])
tspan = [0, 60]
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