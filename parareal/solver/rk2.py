import numpy as np

def rk2(odefun, tspan, y0, Nh, *args):
    """
    Solve differentiel equation using runge_kutta2 method.

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
        u[i+1, :] = u[i, :] + h * odefun(t[i] + h / 2., u[i, :] + odefun(t[i], u[i, :], *args) * h / 2., *args)
    return t,u