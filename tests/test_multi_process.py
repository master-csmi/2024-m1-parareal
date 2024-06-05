import subprocess

# Define the command to run the MPI script
command = ["mpiexec", "-n", "2", "python", "test_mpi_parareal.py"]

# Execute the command
try:
    subprocess.run(command, capture_output=True, text=True)
except:
    print("Error when executing test_mpi_parareal in parallel")

