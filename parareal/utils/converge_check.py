import numpy as np
import matplotlib.pyplot as plt
from parareal.parareal.mpi_parareal import *

def convergence_check(G,F,Ns,tspan,y0,max_iter,tol,f_exacte):
    
    dts=[]
    errors = []

    for n in Ns:
        t, iterations, sol = parareal(G, F, tspan, y0, n, max_iter, tol)
        dts .append(t[1]-t[0])
        y_exac =  np.reshape(f_exacte(t),np.array(sol).shape)
        errors.append(np.linalg.norm(sol - y_exac) / np.linalg.norm(y_exac))
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        
        # Calculate convergence rates
        errors = np.array(errors)
        dts = np.array(dts)
        convergence_rates = np.log(errors[:-1] / errors[1:]) / np.log(dts[:-1] / dts[1:])

        # Calculate the average order of convergence
        order_of_convergence = np.mean(convergence_rates)
        print("Time steps:", dts)
        print("Errors:", errors)
        print("Convergence rates:", convergence_rates)
        print("Order of convergence:", order_of_convergence)
        
    return dts, errors, convergence_rates, order_of_convergence

