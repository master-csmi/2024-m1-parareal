import numpy as np

def parareal(G, F, tspan, y0, N, K, tol = 0.5):
    """
    Parareal Algorithme  for solving differential equations.

    :param T: array of time (start time, end time)
    :param N: Number of sub interval
    :param G: Coarse Function
    :param F: Fine Function
    :param y0: initial Condition 
    :param K: Maximum number of parareal iteration
    :return: t, number of iterations, Approximate solution y at discretization points
    """
    
    times = np.linspace(tspan[0], tspan[1], N + 1)
    
    # Zeros iteration
    u = [y0]
    for i in range(N):
        u.append(G([times[i], times[i + 1]], u[i]))
        
    u = np.array(u)
    G_ = u 
    F_ = np.zeros_like(u)
    F_[0] = y0
    
    for iter in range(K):
        # Copy the previous Solution U and Coarse propagation for parareal correction
        u_prev = u.copy()
        G_prev = G_.copy()
        
        # Fine propagation
        for i in range(N):
            F_[i+1] = F([times[i], times[i + 1]], u[i])
        
        # Coarse propagation and update the solution
        for i in range(N):
            G_[i+1] = G([times[i], times[i + 1]], u[i])
            u[i+1] = G_[i+1] + F_[i+1] - G_prev[i+1]

        # Check the stopping criterion
        error = np.linalg.norm(u - u_prev)/len(times)
        print(f"Iteration {iter}, Error: {error}")
        if error < tol:
            break

    return times, iter + 1, u
