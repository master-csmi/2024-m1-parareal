#!/bin/bash

# Remove old execution times file if it exists
rm -f execution_times.txt

# List of solvers
solvers=("explicit_euler" "implicit_euler" "explicit_rk2" "implicit_rk2" "explicit_rk4" "implicit_rk4")

# Iterate over the number of processes
for np in 1 2 4 8 16 32; do
    # Iterate over all combinations of solvers
    for G_solver in "${solvers[@]}"; do
        for F_solver in "${solvers[@]}"; do
            # Run the Python script with the specified number of processes and solvers
            mpiexec -n $np python lorenz_parallel.py $G_solver $F_solver
        done
    done
done
