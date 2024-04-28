#!/bin/bash
#SBATCH --gpus=rtx_3090:1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=./output_preprocess_facescape.txt
#SBATCH --error=./error_preprocess_facescape.txt
#SBATCH --job-name=preprocess_facescape

# load your openmpi modules
# module load openmpi/4.1.4

mpirun -np 8 python process_all_mpi.py --input_dir /cluster/scratch/xiychen/data/facescape_raw/ --output_dir /cluster/scratch/xiychen/data/facescape_color_calibrated/