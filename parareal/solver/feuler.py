import numpy as np
from scipy.optimize import fsolve

"""
The feuler code is from Tan course of strasboug university (tan.ode code) 
"""
def feuler(odefun, tspan, y0, Nh, *args):
    """
    Résout une équation différentielle en utilisant la méthode d'Euler explicite.

    Paramètres :
    - odefun : La fonction de l'équation différentielle y' = f(t, y, *args)
    - tspan : Tuple (T0, TF) définissant l'intervalle de temps de l'intégration
    - y0 : Condition initiale
    - Nh : Nombre d'intervalles de temps
    - *args : Arguments supplémentaires à passer à odefun

    Retour :
    - t : Vecteur des temps où la solution est calculée
    - u : Solution de l'équation différentielle à chaque instant t
    """
    h = (tspan[1] - tspan[0]) / Nh  # Taille de chaque intervalle
    t = np.linspace(tspan[0], tspan[1], Nh+1)  # Points de temps
    u = np.zeros((Nh+1, len(y0)))  # Initialisation du tableau de solutions
    u[0, :] = y0  # Définir la condition initiale

    for i in range(Nh):
        u[i+1, :] = u[i, :] + h * np.asarray(odefun(t[i], u[i, :], *args))
    
    return t, u