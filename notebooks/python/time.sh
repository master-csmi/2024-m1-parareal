#!/bin/bash

# Remove old execution times file if it exists
rm -f execution_times.txt

# List of solvers
solvers=("explicit_euler" "implicit_euler" "explicit_rk2" "implicit_rk2" "explicit_rk4" "implicit_rk4")

# Iterate over the number of processes
for np in 1 2 4 8 16 32 64 128; do
    # Iterate over all combinations of solvers
    for G_solver in "${solvers[@]}"; do
        for F_solver in "${solvers[@]}"; do

cat > parareal-$G_solver-$F_solver-$np.job <<EOF
#!/bin/bash
#SBATCH -J parareal      # name of the job
#SBATCH -N 1          # number of nodes
#SBATCH --ntasks-per-node=$np # number of MPI processes
#SBATCH --exclusive
#SBATCH -t 0:10:00
#SBATCH --threads-per-core=1   # no hyperthreading

#SBATCH --output=run-parareal-$np-$G_solver-$F_solver.out
#SBATCH --error=run-parareal-$np-$G_solver-$F_solver.err

export OMP_NUM_THREADS=1
module load hpcx


mpiexec -bind-to core python lorenz_parallel.py $G_solver $F_solver
EOF

            # Submit the job
            sbatch parareal-$G_solver-$F_solver-$np.job

            # Wait for the job to finish
            while [ ! -f execution_times.txt ]; do
                sleep 1
            done

            # Append the execution times to the file
            echo "np=$np, G_solver=$G_solver, F_solver=$F_solver" >> execution_times.txt
            cat execution_times.txt >> execution_times.txt

            # Remove the file
            rm -f execution_times.txt
        done
    done
done
