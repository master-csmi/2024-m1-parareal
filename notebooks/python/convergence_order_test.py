import numpy as np
import sys
from mpi4py import MPI
from parareal.solver.euler import feuler
from parareal.solver.beuler import beuler
from parareal.solver.rk2 import rk2
from parareal.solver.brk2 import brk2
from parareal.solver.rk4 import rk4
from parareal.solver.brk4 import brk4
from parareal.utils.converge_check import convergence_check

def get_function(name):
    if name == "f":
        from parareal.utils.test_func import f as func, f_exacte as exact_func
        y0 = np.array([2]) 
    elif name == "z":
        from parareal.utils.test_func import z as func, z_exacte as exact_func
        y0 = np.array([0])
    elif name == "h":
        from parareal.utils.test_func import h as func, h_exacte as exact_func
        y0 = np.array([1])
    else:
        raise ValueError(f"Unknown function name: {name}")
    return func, exact_func, y0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convergence_order_test.py <function_name>")
        sys.exit(1)

    function_name = sys.argv[1]
    func, exact_func, y0 = get_function(function_name)

    tspan = [0, 5] 
    
    tol = 1e-10
    max_iter = 10
    G_Nh=1
    F_Nh=5

    solvers = {
        'explicit_euler': feuler,
        'implicit_euler': beuler,
        'explicit_rk2': rk2,
        'implicit_rk2': brk2,
        'explicit_rk4': rk4,
        'implicit_rk4': brk4,
    }

    Ns = np.arange(20, 101, 20)
    results = {}
    for coarse_name, coarse_solver in solvers.items():
        for fine_name, fine_solver in solvers.items():
            print(f"Running: {coarse_solver.__name__} as coarse and {fine_solver.__name__} as fine solver")
            dts, errors, convergence_rates, order_of_convergence = convergence_check(
                lambda tspan, u0: coarse_solver(func, tspan, u0, G_Nh)[1][-1],
                lambda tspan, u0: fine_solver(func, tspan, u0, F_Nh)[1][-1],
                Ns, tspan, y0, max_iter, tol, exact_func)
            results[(coarse_name, fine_name)] = order_of_convergence

    if MPI.COMM_WORLD.Get_rank() == 0 :
        with open(f"data-cv_order-{function_name}.txt", "a") as f:
            for key, value in results.items():
                f.write(f"{function_name} {key[0]} {key[1]} {value}\n")
