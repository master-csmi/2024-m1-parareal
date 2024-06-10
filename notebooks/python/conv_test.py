import numpy as np
import matplotlib.pyplot as plt
from parareal.parareal.mpi_parareal import parareal
from scipy.integrate import solve_ivp
from parareal.solver.beuler import beuler
from parareal.solver.feuler import feuler
from parareal.solver.rk4 import rk4
from parareal.lorenz.lorenz import lorenz
from parareal.test_func import *
from mpi4py import MPI

def convergence_test(F,G,tspan,y0,Ns,max_iter, f_exacte):
    Ns = np.array([1000,10000])
    dts=[]
    errors = []
    
    for n in Ns:
        t, _, sol1 = parareal(G, F, tspan, y0, n, max_iter,tol=1e-1)
        y_exac =  f_exacte(tspan)[1]
        t_exact = f_exacte(tspan)[0]
        x_sol_exacte_interp = np.interp(t, t_exact, y_exac[:,0])
        y_sol_exacte_interp = np.interp(t, t_exact, y_exac[:,1])
        z_sol_exacte_interp = np.interp(t, t_exact, y_exac[:,2])
        sol_exacte_interp= np.stack((x_sol_exacte_interp, y_sol_exacte_interp, z_sol_exacte_interp), axis=-1)
        errors.append(np.linalg.norm(sol1 - sol_exacte_interp) / len(t))
        dts.append(t[1]-t[0])
    
    errors = np.array(errors)
    dts = np.array(dts)
    
    
    convergence_rates = np.log(errors[:-1] / errors[1:]) / np.log(dts[:-1] / dts[1:])
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
tspan = [0, 4] 
y0 = [2] 
y0 = np.array([1.,1.,1.])
tol = 1e-1
max_iter = 100
G_Nh=1
F_Nh=10

# params
tspan = np.array([0.,60.])
sigma, rho, beta = 10, 28, 8/3

lorenz_sys = lambda _, state :lorenz(_, state, sigma, rho, beta)

G1 = lambda tspan,u0, :beuler(lorenz_sys, tspan, u0, G_Nh)[1][-1]
F1 = lambda tspan,u0, :feuler(lorenz_sys, tspan, u0, F_Nh)[1][-1]

G2 = lambda tspan,u0, :solve_ivp(lorenz_sys, tspan, u0, method='RK23').y[:, -1]
F2 = lambda tspan,u0, :solve_ivp(lorenz_sys, tspan, u0, method='RK45').y[:, -1]

f_exacte = lambda tspan: rk4(lorenz_sys, tspan, y0, Nh)

convergence_test(F1,G1,tspan,y0,Nh,max_iter, f_exacte)
convergence_test(F2,G2,tspan,y0,Nh,max_iter, f_exacte)

