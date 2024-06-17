import numpy as np
import matplotlib.pyplot as plt
from parareal.parareal.parareal import parareal
from scipy.integrate import solve_ivp
from parareal.solver.beuler import beuler
from parareal.solver.feuler import feuler
from parareal.solver.rk4 import rk4
from parareal.solver.rk2 import rk2
from parareal.lorenz.lorenz import lorenz
from parareal.test_func import *

def coarse_solver(tspan, y0):
    """
    Example coarse solver (Euler method).
    """
    dt = tspan[1] - tspan[0]
    return y0 + dt * (-y0)  # Example: dy/dt = -y

def fine_solver(tspan, y0):
    """
    Example fine solver (RK4 method).
    """
    dt = (tspan[1] - tspan[0]) / 100  # Fine step size
    y = y0
    for _ in range(100):
        k1 = dt * (-y)
        k2 = dt * (-(y + 0.5 * k1))
        k3 = dt * (-(y + 0.5 * k2))
        k4 = dt * (-(y + k3))
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y
Nh = 100000
tspan = [0, 10] 
y0 = [1] 
tol = 1e-1
max_iter = 100
G_Nh=1
F_Nh=10



G1 = lambda tspan,u0, :beuler(h, tspan, u0, G_Nh)[1][-1]
F1 = lambda tspan,u0, :rk4(h, tspan, u0, F_Nh)[1][-1]
def test_convergence_rate(tspan, y0, N_range, K, tol):
    """
    Test the convergence rate of the Parareal algorithm.

    :param tspan: Array of time (start time, end time)
    :param y0: Initial condition
    :param N_range: Range of values for the number of sub-intervals
    :param K: Number of Parareal iterations
    :param tol: Tolerance for stopping criterion
    :return: List of (N, iterations) tuples
    """
    
    results = []
    for N in N_range:
        times, iterations, solution = parareal(G1, F1, tspan, y0, N, K, tol)
        results.append((N, iterations))
    
    return results

def plot_convergence_rate(results, filename="convergence_rate.png"):
    """
    Plot the convergence rate of the Parareal algorithm.

    :param results: List of (N, iterations) tuples
    :param filename: Name of the file to save the plot
    """
    
    N_values, iterations = zip(*results)
    plt.plot(N_values, iterations, marker='o')
    plt.xlabel('Number of sub-intervals (N)')
    plt.ylabel('Number of iterations')
    plt.title('Convergence Rate of Parareal Algorithm')
    plt.grid(True)
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

# Parameters
tspan = [0, 1]
y0 = np.array([1.0])
N_range = range(100, 500, 100)  # Varying number of sub-intervals
K = 100
tol = 1e-15

# Test convergence rate
results = test_convergence_rate(tspan, y0, N_range, K, tol)

# Plot the results and save to a file
plot_convergence_rate(results)
