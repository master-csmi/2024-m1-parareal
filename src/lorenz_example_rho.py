from parareal import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from lorenz import *

# solutions of the Lorenz system for different values of rho
# Parameters

# Model parameters
U0 = np.array([1.0, 1.0, 1.0])
t_span = [0, 30]
N = 2000
tol = 1e-1
max_iter = 5
G_Nh=1
F_Nh=10

params =[(10.0, 13.0, 8/3),(10.0, 14.0, 8/3),(10.0, 15.0, 8/3),(10.0, 28.0, 8/3)]

fig = plt.figure(figsize=(12, 12))
for i in range(4):
    sigma, rho, beta = params[i]
    print(sigma, rho, beta)
    lorenz_ = lambda t,state:lorenz(t, state, sigma, rho, beta)
    G = lambda tspan,u0, :solve_ivp(lorenz_, tspan, u0, method='RK23').y[:, -1]
    F = lambda tspan,u0, :solve_ivp(lorenz_, tspan, u0, method='RK23').y[:, -1]
    # Run Parareal
    t, iterations, sol = parareal(G, F, t_span, U0, N, max_iter, tol)
    
    
    ax = fig.add_subplot(221 + i,projection='3d')
    ax.plot(sol[:,0], sol[:,1], sol[:,2], lw=0.5)
    ax.set_title('Default View')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
plt.show()