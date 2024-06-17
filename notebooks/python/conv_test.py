import numpy as np
import matplotlib.pyplot as plt
from parareal.parareal.parareal import parareal
from scipy.integrate import solve_ivp
from parareal.solver.beuler import beuler
from parareal.solver.feuler import feuler
from parareal.solver.rk4 import rk4
from parareal.solver.rk2 import rk2
from parareal.lorenz.lorenz import lorenz
from parareal.test_func import *
from mpi4py import MPI

def convergence_test(F,G,tspan,y0,Nh,max_iter, f_exacte):
    Ns = np.array([10,100,1000])
    dts=[]
    errors = []
    
    for n in Ns:
        t, _, sol1 = parareal(G, F, tspan, y0, n, max_iter,tol=1e-10)
        sol_exacte =  f_exacte(t)
        errors.append(np.linalg.norm(sol1-sol_exacte))
        dts.append(t[1]-t[0])
    
    errors = np.array(errors)
    dts = np.array(dts)
    
    
    convergence_rates = ((errors[:-1] / errors[1:])) / ((dts[:-1] / dts[1:]))
    order_of_convergence = np.mean(convergence_rates)
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Time steps:", dts)
        print("Errors:", errors)
        print("Convergence rates:", convergence_rates)
        print("Order of convergence:", order_of_convergence)

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(221)
        ax1.loglog(dts, errors, '-o')
        ax1.set_xlabel('Time step $\Delta t$')
        ax1.set_ylabel('Error')
        ax1.set_title('Test using implicit euler')
        plt.show()
    
    return dts, errors, convergence_rates, order_of_convergence

Nh = 100000
tspan = [0, 100] 
y0 = [1] 
tol = 1e-1
max_iter = 100
G_Nh=1
F_Nh=10



G1 = lambda tspan,u0, :rk4(h, tspan, u0, G_Nh)[1][-1]
F1 = lambda tspan,u0, :rk4(h, tspan, u0, F_Nh)[1][-1]

G2 = lambda tspan,u0, :solve_ivp(h, tspan, u0, method='RK23').y[:, -1]
F2 = lambda tspan,u0, :solve_ivp(h, tspan, u0, method='RK45').y[:, -1]

convergence_test(F1,G1,tspan,y0,Nh,max_iter, h_exacte)
#convergence_test(F2,G2,tspan,y0,Nh,max_iter, h_exacte)

# Model parameters
sigma, rho, beta = 10.0, 28.0, 8/3
U0 = np.array([5.0, -5.0, 20.0])
tspan = [0, .1]
lorenz_ = lambda t,state:lorenz(t, state, sigma, rho, beta)

#Parareal params
N = 180
tol = 1e-10
max_iter = 75
G_Nh=1
F_Nh=1
G3 = lambda tspan,u0, :beuler(lorenz_, tspan, u0, G_Nh)[1][-1]
F3 = lambda tspan,u0, :rk4(lorenz_, tspan, u0, F_Nh)[1][-1]

def lorenz_ref(t):
    sol = solve_ivp(lorenz_, tspan, U0, method='RK45')
    sol_reference = sol.y.T
    t_reference = sol.t
    # Interpolation de la solution de référence pour correspondre aux temps calculés
    x = np.interp(t, t_reference, sol_reference[:,0])
    y = np.interp(t, t_reference, sol_reference[:,1])
    z = np.interp(t, t_reference, sol_reference[:,2])
    return np.stack((x,y,z),axis=1)

#convergence_test(F3,G3,tspan,U0,Nh,max_iter, lorenz_ref)



