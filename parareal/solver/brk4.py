import numpy as np
from scipy.optimize import fsolve

def brk4(f, tspan, y0, Nh, *args):
    """
    Implementation of the fourth-order implicit Runge-Kutta Gauss-Legendre method.

    :param f: Function defining the differential equation dy/dt = f(t, y)
    :param tspan: Tuple containing the initial and final time (t0, tf)
    :param y0: Initial condition as an array
    :param Nh: Number of time intervals
    :param args: Additional arguments for the function f
    :return: Tuple (times, y) where times is the array of time points and y is the array of solutions
    """
    h = (tspan[1] - tspan[0]) / Nh
    times = np.linspace(tspan[0], tspan[1], Nh + 1)
    y = np.zeros((Nh + 1, len(y0)))
    y[0] = y0

    # Coefficients for the fourth-order Gauss-Legendre method
    sqrt_3 = np.sqrt(3) / 6
    c = np.array([0.5 - sqrt_3, 0.5 + sqrt_3])
    A = np.array([
        [0.25, 0.25 - sqrt_3],
        [0.25 + sqrt_3, 0.25]
    ])
    b = np.array([0.5, 0.5])
    s = len(c)  # Number of stages

    def gauss_legendre_fun(Y, yi, ti, h):
        Y = Y.reshape(s, -1)
        F = np.array([f(ti + c[i] * h, Y[i], *args) for i in range(s)])
        G = Y - yi - h * (A @ F)
        return G.flatten()

    for i in range(1, Nh + 1):
        ti = times[i - 1]
        yi = y[i - 1]
        Y_guess = np.tile(yi, s)
        Y_sol = fsolve(gauss_legendre_fun, Y_guess, args=(yi, ti, h))
        Y_sol = Y_sol.reshape(s, -1)
        y[i] = yi + h * np.sum(b[j] * f(ti + c[j] * h, Y_sol[j], *args) for j in range(s))

    return times, y
