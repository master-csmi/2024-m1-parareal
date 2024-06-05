import numpy as np
from scipy.optimize import fsolve

"""
The beuler code is from Tan course of strasboug university (tan.ode code) 
"""

def beuler(edofon, tspan, y0, Nh, *args):
    """
    Résout une équation différentielle en utilisant la méthode d'Euler arrière.

    Paramètres :
    - edofon : fonction de l'EDO y'=f(t,y)
    - tspan : Tuple (T0, TF) pour l'intervalle de temps
    - y0 : condition initiale
    - Nh : nombre d'intervalles
    - *args : arguments supplémentaires pour edofon
    """
    h = (tspan[1] - tspan[0]) / Nh
    tt = np.linspace(tspan[0], tspan[1], Nh+1)
    y = y0
    u = [y0]

    for t in tt[1:]:
        # Définition de la fonction pour fsolve
        def beulerfun(w):
            return w - y - h * edofon(t, w, *args)
        
        # Utilisation de fsolve pour trouver w
        w, info, ier, mesg = fsolve(beulerfun, y, full_output=True)
        if ier != 1:
            raise ValueError(f"fsolve n'a pas convergé : {mesg}")
        u.append(w)
        y = w

    return tt, np.array(u)