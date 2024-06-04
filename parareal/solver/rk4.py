import numpy as np

def rk4(odefun, tspan, y0, Nh, *args):
    """
    Solve differentiel equation using runge_kutta4 method.

    Params :
    - odefun : differentiel equation
    - tspan : time interval
    - y0 : initial condition
    - Nh : number of step size
    - *args : Arguments of odefun

    Return :
    - t : array of time
    - u : Solution of the differentiel equation at each t
    """
    h = (tspan[1] - tspan[0]) / Nh 
    t = np.linspace(tspan[0], tspan[1], Nh+1) 
    u = np.zeros((Nh+1, len(y0))) 
    u[0, :] = y0 

    for i in range(Nh):
        k1 = odefun(t[i], u[i, :], *args)
        k2 = odefun(t[i] + h / 2., u[i, :] + k1 * h / 2., *args)
        k3 = odefun(t[i] + h / 2., u[i, :] + k2 * h / 2., *args)
        k4 = odefun(t[i] + h, u[i, :] + k3 * h, *args)
        u[i+1, :] = u[i, :] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, u