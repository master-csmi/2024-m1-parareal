import numpy as np

def parareal(G, F, tspan, y0, N, K, tol = 0.5):
    """
    Algorithme Parareal pour résoudre un système d'équations différentielles.

    :param T: array of time (start time, end time)
    :param N: Nombre de sous-intervalles
    :param G: Fonction grossière
    :param F: Fonction fine
    :param y0: Condition initiale
    :param K: Nombre d'itérations Parareal
    :return: t, nombre d'iterations, Solution approximative y aux points de discrétisation
    """
    
    times = np.linspace(tspan[0], tspan[1], N + 1)
    
    u = [y0]
    for i in range(N):
        u.append(G([times[i], times[i + 1]], u[i]))
        
    u = np.array(u)
    G_ = u 
    F_ = u
    
    for iter in range(K):
        u_prev = u.copy()
        G_prev = G_.copy()
        
        for i in range(N):
            F_[i+1] = F([times[i], times[i + 1]], u[i])
            
        for i in range(N):
            G_[i+1] = G([times[i], times[i + 1]], u[i])
            u[i+1] = G_[i+1] + F_[i+1] - G_prev[i+1]

        # Check the stopping criterion
        if np.linalg.norm(u - u_prev) < tol:
            break

    return times,iter, u

