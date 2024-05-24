import numpy as np

def rungekutta2(odefun, tspan, y0, Nh, *args):
    
    h = (tspan[1] - tspan[0]) / Nh  # Taille de chaque intervalle
    t = np.linspace(tspan[0], tspan[1], Nh+1)  # Points de temps
    u = np.zeros((Nh+1, len(y0)))  # Initialisation du tableau de solutions
    u[0, :] = y0  # Définir la condition initiale
    
    for i in range(Nh):
        u[i+1, :] = u[i, :] + h * odefun(t[i] + h / 2., u[i, :] + odefun(t[i], u[i, :], *args) * h / 2., *args)
    return t,u

def rungekutta4(odefun, tspan, y0, Nh, *args):

    h = (tspan[1] - tspan[0]) / Nh  # Taille de chaque intervalle
    t = np.linspace(tspan[0], tspan[1], Nh+1)  # Points de temps
    u = np.zeros((Nh+1, len(y0)))  # Initialisation du tableau de solutions
    u[0, :] = y0  # Définir la condition initiale

    for i in range(Nh):
        k1 = odefun(t[i], u[i, :], *args)
        k2 = odefun(t[i] + h / 2., u[i, :] + k1 * h / 2., *args)
        k3 = odefun(t[i] + h / 2., u[i, :] + k2 * h / 2., *args)
        k4 = odefun(t[i] + h, u[i, :] + k3 * h, *args)
        u[i+1, :] = u[i, :] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, u

def adaptive_rungekutta4(odefun, tspan, y0, hmax , tol = 1e-2, *args):
    
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