#!/bin/bash

# Remove old execution times file if it exists
rm -f execution_times.txt job_ids.txt

# List of solvers
solvers=("explicit_euler" "implicit_euler" "explicit_rk2" "implicit_rk2" "explicit_rk4" "implicit_rk4")

# Maximum number of jobs to submit at once
max_jobs=10

# Function to check and wait if the number of running jobs exceeds max_jobs
wait_for_jobs() {
    while [ $(squeue -u $USER | grep 'parareal' | wc -l) -ge $max_jobs ]; do
        sleep 5
    done
}

# Iterate over the number of processes
for np in 1 2 4 8 16 32 64 128; do
    # Iterate over all combinations of solvers
    for G_solver in "${solvers[@]}"; do
        for F_solver in "${solvers[@]}"; do
            cat > parareal-$G_solver-$F_solver-$np.job <<EOF
#!/bin/bash
#SBATCH -J parareal      # name of the job
#SBATCH -N 1             # number of nodes
#SBATCH --ntasks-per-node=$np # number of MPI processes
#SBATCH --exclusive
#SBATCH -t 0:20:00
#SBATCH --threads-per-core=1   # no hyperthreading

#SBATCH --output=run-parareal-$np-$G_solver-$F_solver.out
#SBATCH --error=run-parareal-$np-$G_solver-$F_solver.err

export OMP_NUM_THREADS=1

mpiexec -bind-to core python lorenz_parallel.py $G_solver $F_solver
EOF

            # Submit the job and record the job ID
            job_id=$(sbatch parareal-$G_solver-$F_solver-$np.job | awk '{print $4}')
            echo $job_id >> job_ids.txt
            wait_for_jobs
        done
    done
done

# Wait for all remaining jobs to finish
while squeue -u $USER | grep -q 'parareal'; do
    sleep 1
done

# Concatenate all execution times
for np in 1 2 4 8 16 32 64 128; do
    for G_solver in "${solvers[@]}"; do
        for F_solver in "${solvers[@]}"; do
            if [[ -f execution-times-$np-$G_solver-$F_solver.txt ]]; then
                cat execution-times-$np-$G_solver-$F_solver.txt >> execution_times.txt
                rm execution-times-$np-$G_solver-$F_solver.txt
            fi
        done
    done
done

# Cleanup: Remove job and output files
rm -f parareal-*.job
rm -f run-parareal-*.out
rm -f run-parareal-*.err
