from parareal.parareal.mpi_parareal import mpi_parareal
from parareal.solver.feuler import feuler
from parareal.utils.test_func import *
from mpi4py import MPI
import numpy as np

def test_mpi_parareal():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    Nh = 100 
    tspan = [0, 4] 
    y0 = [2] 

    if rank == 0:
        print("Starting MPI Parareal Test...")

    t = np.linspace(0, 4, 100)
    y = f_exacte(t)

    tol = 1e-10
    max_iter = 10
    G_Nh = 1
    F_Nh = 10
    G = lambda tspan, u0: feuler(f, tspan, u0, G_Nh)[1][-1]
    F = lambda tspan, u0: feuler(f, tspan, u0, F_Nh)[1][-1]

    try:
        t_sol, iterations, solution, time = mpi_parareal(G, F, tspan, y0, Nh, max_iter, tol)
        if rank == 0:
            print("MPI Parareal completed.")

        y_exac = np.reshape(f_exacte(t), np.array(solution[-1]).shape)
        error = np.linalg.norm(solution[-1] - y_exac) / np.linalg.norm(y_exac)
        if rank == 0:
            print(f"Error: {error}")
        assert error <= 1e-8
        
    except Exception as e:
        if rank == 0:
            print(f"Error when executing parareal in Parallel function: {e}")

    finally:
        if rank == 0:
            print("Finalizing MPI")
        MPI.Finalize()

if __name__ == "__main__":
    test_mpi_parareal()
