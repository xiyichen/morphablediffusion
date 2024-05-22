#!/bin/bash
#SBATCH --gpus=1
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --output=./output_preprocess_thuman.txt
#SBATCH --error=./error_preprocess_thuman.txt
#SBATCH --job-name=preprocess_thuman

# load your openmpi and blender modules
module load openmpi/4.1.4
module load blender/3.4.1

mpirun -np 8 python render_batch_mpi.py --input_dir /cluster/scratch/xiychen/data/thuman_2.1 --output_dir /cluster/scratch/xiychen/data/thuman_2.1_preprocessed