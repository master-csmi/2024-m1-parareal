from parareal.test_func import *
from scipy.integrate import solve_ivp
from parareal.parareal.mpi_parareal import *
from parareal.solver.beuler import beuler
import matplotlib.pyplot as plt
from mpi4py import MPI
# Model parameters


Nh = 100000
tspan = [0, 4] 
y0 = [2] 
tol = 1e-1
max_iter = 10
G_Nh=1
F_Nh=1

G1 = lambda tspan,u0, :beuler(f, tspan, u0, G_Nh)[1][-1]
F1 = lambda tspan,u0, :beuler(f, tspan, u0, F_Nh)[1][-1]

G2 = lambda tspan,u0, :solve_ivp(f, tspan, u0, method='RK23').y[:, -1]
F2 = lambda tspan,u0, :solve_ivp(f, tspan, u0, method='RK45').y[:, -1]

# Run Parareal
#t_reference, iterations, sol_reference = parareal(G, F, tspan, y0, Nh, max_iter, tol)
#dt_reference = t_reference[1]-t_reference[0]


Ns = np.array([10,100,1000])
dts=[]
errors1 = []
errors2 = []

for n in Ns:
    t, iterations, sol1 = parareal(G1, F1, tspan, y0, n, max_iter, tol)
    t, iterations, sol2 = parareal(G2, F2, tspan, y0, n, max_iter, tol)
    dts .append(t[1]-t[0])
    # Interpolation de la solution de référence pour correspondre aux temps calculés
    #sol_reference_interp = np.interp(t, t_reference, sol_reference[:,0])
    y_exac =  f_exacte(t)
    errors1.append(np.linalg.norm(sol1[:,0] - y_exac) / len(t))
    errors2.append(np.linalg.norm(sol2[:,0] - y_exac) / len(t))
    

if MPI.COMM_WORLD.Get_rank() == 0:
    
    # Calculate convergence rates
    errors1 = np.array(errors1)
    dts = np.array(dts)
    convergence_rates = np.log(errors1[:-1] / errors1[1:]) / np.log(dts[:-1] / dts[1:])

    # Calculate the average order of convergence
    order_of_convergence = np.mean(convergence_rates)
    print("Time steps:", dts)
    print("Errors:", errors1)
    print("Convergence rates:", convergence_rates)
    print("Order of convergence:", order_of_convergence)
    
    errors2 = np.array(errors2)
    convergence_rates = np.log(errors2[:-1] / errors2[1:]) / np.log(dts[:-1] / dts[1:])

    # Calculate the average order of convergence
    order_of_convergence = np.mean(convergence_rates)
    print("Time steps:", dts)
    print("Errors:", errors2)
    print("Convergence rates:", convergence_rates)
    print("Order of convergence:", order_of_convergence)

    # Affichage des résultats de convergence
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(221)
    ax1.loglog(dts, errors1, '-o')
    ax1.set_xlabel('Time step $\Delta t$')
    ax1.set_ylabel('Error')
    ax1.set_title('Test using implicit euler')
    
    ax2 = fig.add_subplot(222)
    ax2.loglog(dts, errors2, '-o')
    ax2.set_xlabel('Time step $\Delta t$')
    ax2.set_ylabel('Error')
    ax2.set_title('Test using solve_ivp(Rk23,Rk45)')
    plt.show()
    