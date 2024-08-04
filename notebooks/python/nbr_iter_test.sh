#!/bin/bash

# Check the number of arguments (Number of MPI processes)
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_mpi_processes>"
    echo "No argument provided. Setting the number of MPI processes to 1."
    num_mpi_processes=1
else
    num_mpi_processes=$1
fi

# Get the current date and time
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Define the output directory
output_dir="data"
mkdir -p $output_dir

# Define the functions to test
functions=("f" "z" "h")

# Maximum number of jobs to submit at once
max_jobs=10

# File to store job IDs
job_ids_file="${output_dir}/job_ids_nbr_iter_${timestamp}.txt"

# Function to check and wait if the number of running jobs exceeds max_jobs
wait_for_jobs() {
    while [ $(squeue -u $USER | grep 'parareal' | wc -l) -ge $max_jobs ]; do
        sleep 5
    done
}

# Loop over each function and create and submit the job script
for func in "${functions[@]}"
do
    cat > parareal-$func-$timestamp.job <<EOF
#!/bin/bash
#SBATCH -J parareal-$func # name of the job
#SBATCH -N 1              # number of nodes
#SBATCH --ntasks-per-node=$num_mpi_processes # number of MPI processes
#SBATCH --exclusive
#SBATCH -t 0:10:00
#SBATCH --threads-per-core=1   # no hyperthreading
#SBATCH --output=run-parareal-nbr_iter-${func}.out
#SBATCH --error=run-parareal-nbr_iter-${func}.err

export OMP_NUM_THREADS=1
mpiexec -bind-to core python nbr_iter_test.py $func 
EOF

    # Submit the job and record the job ID
    job_id=$(sbatch parareal-$func-$timestamp.job | awk '{print $4}')
    echo $job_id >> $job_ids_file
    wait_for_jobs
done

# Wait for all remaining jobs to finish
while squeue -u $USER | grep -q 'parareal'; do
    sleep 1
done

# Combine results into a single data file
data_file="${output_dir}/data_nbr_iter_${timestamp}.txt"
for func in "${functions[@]}"
do
    result_file="data-nbr_iter-${func}.txt"
    if [[ -f $result_file ]]; then
        cat $result_file >> $data_file
        rm $result_file
    fi
done

# Cleanup: Remove job and output files
rm -f parareal-*.job
rm -f run-parareal-*.out
rm -f run-parareal-*.err
