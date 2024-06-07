# Parareal Algorithm

## Project Description
This project focuses on the development and implementation of the parareal algorithm in both sequential and parallel formats. The algorithm will be applied to the Lorenz Model of order 1 and 4, using both constant and adaptive time stepping.

## Roadmap
1. Implement Lorenz Model for order 1 and 4 using constant time stepping.
2. Implement Lorenz order 4 solver with adaptive time stepping.
3. Add support for Lorenz results visualisation.
4. Study scipy.ode with constant and adaptive time stepping.
5. Study the parareal algorithm.
6. Implement the initial version of parareal in sequential.
7. Compare different ode solvers and parareal on the Lorenz system.
8. Study a parallel version of the parareal algorithm.
9. Implement the parallel version of the parareal algorithm.

## Getting Started
To get started with this project, clone the repository and install the required dependencies.

## Prerequisites
- Python 3.x
- Scipy
- Matplotlib (for visualization)
- mpi4py (for parallelisation)

## Installation
1. Clone the repo
   ```sh
   git clone https://github.com/master-csmi/2024-m1-parareal.git
2. Installation des paquets python requis

   Pour utiliser les programmes du cours et executer les notebooks jupyter
   , veuillez suivre la procédure suivante:

   ```sh
   python3 -m venv .venv # <1>
   source .venv/bin/activate # <2>
   pip3 install -r requirements.txt # <3>
   ```

   1. Creation d'un environnement virtuel python `.venv`
   2. Activation de l'environnement virtuel `.venv`
   3. Installation des paquets python requis
   
## Usage
To execute MPI4py code in the terminal, you can follow these steps:

1. Ensure that you have installed MPI on your system.

2. Compile your MPI4py code using the following command:
   ```sh
   mpiexec -n <number_of_processes> python your_code.py
   ```
   Replace `<number_of_processes>` with the desired number of MPI processes you want to run.
   Replace `your_code.py` with the source name code you want to run.

## Contributors
- <a href="https://github.com/oussama-floor9" target="blank">Oussama BOUHENNICHE</a>
- <a href="https://github.com/zaouach" target="blank">Narimane ZAOUACHE</a>