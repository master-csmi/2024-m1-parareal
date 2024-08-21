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
2. Installation of required Python packages

   To use the program and execute Jupyter notebooks, please follow the following procedure:

   ```sh
   python3 -m venv .venv # <1>
   source .venv/bin/activate # <2>
   pip3 install -r requirements.txt # <3>
   ```

   1. Create a Python virtual environment `.venv`
   2. Activate the virtual environment `.venv`
   3. Installation of required Python packages
   
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
---
# Internship Continuation
## Internship Goals

The internship aims to extend the initial project by implementing advanced features and performing detailed analyses.

## Extended Roadmap

1. Implement Implicit RK4 and RK2 solvers.
2. Perform Parareal speedup analysis.
3. Study the convergence order of the algorithm for different solver combinations.
4. Study the number of Parareal iterations required for convergence for different solver combinations.

## Usage
Follow the same steps as outlined in the initial project section to run the code.

## Contributors
- <a href="https://github.com/oussama-floor9" target="blank">Oussama BOUHENNICHE</a>

## Running Analyses using Shell Scripts

1. Convergence Order Analysis:
   ```sh
   ./convergence_order_test.sh <number_of_mpi_processes>
   ```
   This script runs the convergence order analysis for different solver combinations.
   Replace `<number_of_processes>` with the desired number of MPI processes.

2. Parareal Iterations Analysis:
   ```sh
   ./nbr_iter_test.sh <number_of_mpi_processes>
   ```
   This script runs the analysis of Parareal iterations required for convergence for different solver combinations.
   Replace `<number_of_processes>` with the desired number of MPI processes.

3. Parareal Speedup Analysis:
   ```sh
   ./speedup_test.sh
   ```
   This script runs the Speedup analysis of Parareal for different solver combinations.