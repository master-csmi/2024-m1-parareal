import numpy as np
from scipy.optimize import fsolve

def brk2(f, tspan, y0, Nh, *args):
    """
    Implémentation de la méthode de Runge-Kutta implicite d'ordre 2 (Gauss-Legendre).

    :param f: fonction définissant l'équation différentielle dy/dt = f(t, y)
    :param y0: condition initiale
    :param t0: temps initial
    :param tf: temps final
    :param args: pas de temps
    :return: tableau des temps et des solutions
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
