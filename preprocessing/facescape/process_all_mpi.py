from mpi4py import MPI
import subprocess
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    flags = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    subject_ids = list(range(1, 360, 1))
    # Divide subject IDs among MPI processes
    chunk_size = len(subject_ids) // size
    subject_ids_chunk = subject_ids[rank * chunk_size : (rank + 1) * chunk_size]
    if rank == size - 1:
        # Ensure the last process gets any remaining subjects
        subject_ids_chunk += subject_ids[size * chunk_size:]
    # Iterate over subject IDs assigned to this process
    for subject_id in subject_ids_chunk:
        # Construct the command to execute
        command = f"python process_dataset.py --dir_in {os.path.join(flags.input_dir, str(subject_id))} --dir_out {os.path.join(flags.output_dir, str(subject_id).zfill(3))} --save_bilinear_vertices True"
        # Execute the command
        subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    main()