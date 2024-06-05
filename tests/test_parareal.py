from parareal.parareal.parareal import parareal
from parareal.solver.feuler import feuler
from parareal.test_func import *

def test_parareal():
    Nh = 100 
    tspan = [0, 4] 
    y0 = [2] 

    # tracons la solution exacte
    t = np.linspace(0, 4, 100)
    y = f_exacte(t)

    tol = 1e-1
    max_iter = 10
    G_Nh=1
    F_Nh=10
    G = lambda tspan,u0, :feuler(f, tspan, u0, G_Nh)[1][-1]
    F = lambda tspan,u0, :feuler(f, tspan, u0, F_Nh)[1][-1]

    try:
        parareal(G, F, tspan, y0, Nh, max_iter, tol)
    except:
        print("Error when executing parareal function")
