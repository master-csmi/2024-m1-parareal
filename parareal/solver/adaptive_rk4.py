import numpy as np

def adaptive_rk4(odefun, tspan, y0, hmax , tol = 1e-2, *args):
    """
    Solve differentiel equation using adaptive_rungekutta4 method.

    Params :
    - odefun : differentiel equation
    - tspan : time interval
    - y0 : initial condition
    - hmax : the maximum value of the step size
    - tol: tolerance
    - *args : Arguments of odefun

    Return :
    - t : array of time
    - u : Solution of the differentiel equation at each t
    """
    def rk4(f, t, y, h):
        k1 = f(t, y, *args)
        k2 = odefun(t + h / 2., y + k1 * h / 2., *args)
        k3 = odefun(t + h / 2., y + k2 * h / 2., *args)
        k4 = odefun(t + h, y + k3 * h, *args)
        return y + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    
    t0, tf = tspan
    h = hmax
    t = [t0]
    y = [y0]

    while t[-1] < tf:
        y_step = rk4(odefun, t[-1], y[-1], h)
        y_double_step = rk4(odefun, t[-1] + h/2, y[-1], h/2)

        #truncation error
        error = np.linalg.norm(y_double_step - y_step) / np.linalg.norm(y_double_step)

        #stepsize control
        if error < tol:
            t.append(t[-1]+h)
            y.append(y_double_step)
            h = min(2*h, hmax)
        else:
            h *= 0.5
    
    return np.array(t), np.array(y)