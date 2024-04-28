import subprocess

from mpi4py import MPI
import subprocess
import argparse

def main(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    subject_ids = list(range(2445))
    subject_ids = [str(subject_id).zfill(4) for subject_id in subject_ids]
    chunk_size = len(subject_ids) // size
    subject_ids_chunk = subject_ids[rank * chunk_size : (rank + 1) * chunk_size]
    if rank == size - 1:
        # Ensure the last process gets any remaining subjects
        subject_ids_chunk += subject_ids[size * chunk_size:]
    # Iterate over subject IDs assigned to this process
    for subject_id in subject_ids_chunk:
        # Construct the command to execute 
        cmds = ['blender','--background','--python','blender_script.py','--',
                '--object_path',f'{args.input_dir}/{subject_id}/{subject_id}.obj',
                '--smplx_stats_path',f'{args.output_dir}/smplx_stats/{subject_id}.npy',
                '--output_dir',f'{args.output_dir}/target','--camera_type','fixed']
        subprocess.run(cmds)
        cmds = ['blender','--background','--python','blender_script.py','--',
                '--object_path',f'{args.input_dir}/{subject_id}/{subject_id}.obj',
                '--smplx_stats_path',f'{args.output_dir}/smplx_stats/{subject_id}.npy',
                '--output_dir',f'{args.output_dir}/input','--camera_type','random']
        subprocess.run(cmds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='/cluster/scratch/xiychen/data/thuman_2.1')
    parser.add_argument("--output_dir", type=str, default='/cluster/scratch/xiychen/data/thuman_2.1_preprocessed')
    args = parser.parse_args()
    main(args)