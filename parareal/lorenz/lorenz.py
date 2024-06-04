import numpy as np

def lorenz(_, state, sigma=10, rho=28, beta=8/3):
    """
    system of three differential equations.
    
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    Params :
    - state : array, shape (3,)
    - sigma, rho, beta : parameters defining the Lorenz system

    Return :
    -  array, shape (3,) Values of the Lorenz system partial derivatives at x, y, z
    """
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x*y - beta*z])