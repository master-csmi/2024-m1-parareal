from mpi4py import MPI
import numpy as np
import time

def mpi_parareal(G, F, tspan, y0, N, K, tol=0.5):
    """
    Parareal algorithm to solve a system of differential equations.

    :param G: Coarse solver function
    :param F: Fine solver function
    :param tspan: Array of time (start time, end time)
    :param y0: Initial condition
    :param N: Number of sub-intervals
    :param K: Number of Parareal iterations
    :param tol: Tolerance for stopping criterion
    :return: Approximate solution y at discretization points, number of iteration k, array time
    """
    start_time = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    times = np.linspace(tspan[0], tspan[1], N + 1)
    
    if rank == 0:
        u = [y0]
        for i in range(N):
            u.append(G([times[i], times[i + 1]], u[i]))
        u = np.array(u)
        
        G_ = u.copy()
    else:
        u = None
    
    u = comm.bcast(u, root=0)
    
    F_ = np.zeros_like(u)
    F_[0] = y0

    solutions = [u.copy()]
    
    for iter in range(K):
        u_prev = u.copy()
        
        # Scatter the time slices among the processes
        local_indices = np.array_split(range(N), size)[rank]
        local_F_ = np.zeros_like(F_)
        
        for i in local_indices:
            local_F_[i+1] = F([times[i], times[i + 1]], u[i])
        
        # Gather the fine solver results from all processes into process 0
        comm.Reduce(local_F_, F_, op=MPI.SUM, root=0)
        F_[0] = y0
        
        # Update the solution 
        if rank == 0:
            G_prev = G_.copy()
            for i in range(N):
                G_[i+1] = G([times[i], times[i + 1]], u[i])
                u[i+1] = G_[i+1] + F_[i+1] - G_prev[i+1]
        
        # Share updated solution to all process for nexr iteration
        u = comm.bcast(u, root=0)
        
        solutions.append(u.copy())
        
        # Check the stopping criterion
        error = np.linalg.norm(u - u_prev) / np.linalg.norm(u)
        if rank == 0:
            print(f"Iteration {iter}, Error: {error}")
        if error < tol:
            break
        
    elapsed_time = time.time() - start_time
    
    return times, iter + 1, np.array(solutions), elapsed_time
