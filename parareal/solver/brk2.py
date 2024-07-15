import numpy as np
from scipy.optimize import fsolve

def brk2(f, tspan, y0, Nh, *args):
    """
    Implementation of the seconde-order implicit Runge-Kutta Gauss-Legendre method.

    Params :
    - f: Function defining the differential equation dy/dt = f(t, y)
    - tspan: Tuple containing the initial and final time (t0, tf)
    - y0: Initial condition as an array
    - Nh: Number of time intervals
    - *args: Additional arguments for the function f

    Return :
    - t : array of time
    - u : Solution of the differentiel equation at each t
    """
    
    def phi(y_guess, yi, ti, h):
        return y_guess - yi - h * f(ti + h / 2, (yi + y_guess) / 2, *args)

    h = (tspan[1] - tspan[0]) / Nh
    times = np.linspace(tspan[0], tspan[1], Nh+1)
    y = np.zeros((len(times), len(y0)))
    y[0] = y0

    for i in range(1, len(times)):
        ti = times[i-1]
        yi = y[i-1]
        y_guess = yi + h * f(ti, yi, *args)  # initial guess for solver
        y[i] = fsolve(phi, y_guess, args=(yi, ti, h))
    
    return times, y
