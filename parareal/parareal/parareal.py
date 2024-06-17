import numpy as np

def parareal(G, F, tspan, y0, N, K, tol=1e-6):
    """
    Parareal algorithm to solve a system of differential equations sequentially.

    :param G: Coarse solver function
    :param F: Fine solver function
    :param tspan: Array of time (start time, end time)
    :param y0: Initial condition
    :param N: Number of sub-intervals
    :param K: Number of Parareal iterations
    :param tol: Tolerance for stopping criterion
    :return: Approximate solution y at discretization points, number of iteration k, array time
    """
    
    times = np.linspace(tspan[0], tspan[1], N + 1)
    
    # Initial coarse solution
    u = [y0]
    for i in range(N):
        u.append(G([times[i], times[i + 1]], u[i]))
    u = np.array(u)
    
    G_ = u.copy()
    F_ = np.zeros_like(u)
    F_[0] = y0
    
    for iter in range(K):
        u_prev = u.copy()
        
        # Compute fine solution
        for i in range(N):
            F_[i + 1] = F([times[i], times[i + 1]], u[i])
        
        # Update the solution
        G_prev = G_.copy()
        for i in range(N):
            G_[i + 1] = G([times[i], times[i + 1]], u[i])
            u[i + 1] = G_[i + 1] + F_[i + 1] - G_prev[i + 1]
        
        # Check the stopping criterion
        error = np.linalg.norm(u - u_prev)
        print(f"Iteration {iter}, Error: {error}")
        if error < tol:
            break
    
    return times, iter + 1, u

# Example usage (you need to define G, F, tspan, y0, N, K, tol)
# G = ...
# F = ...
# tspan = [0, 1]
# y0 = np.array([1.0])
# N = 10
# K = 5
# tol = 0.01
# times, iterations, solution = parareal(G, F, tspan, y0, N, K, tol)
