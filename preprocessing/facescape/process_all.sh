#!/bin/bash

# Define the number of CPU cores to use
NUM_CORES=8

export input_dir=/cluster/scratch/xiychen/data/fsmview_trainset
export output_dir=/cluster/scratch/xiychen/data/facescape_color_calibrated_new

# Define the list of subject IDs
SUBJECT_IDS=(1 359)

# Calculate the number of subjects per process
TOTAL_SUBJECTS=${SUBJECT_IDS[1]}
SUBJECTS_PER_PROCESS=$((($TOTAL_SUBJECTS + $NUM_CORES - 1) / $NUM_CORES))

# Function to process a range of subject IDs
process_subjects() {
    local start=$1
    local end=$2
    for ((id=start; id<=end; id++)); do
        echo "Processing subject $id"
        python process_dataset.py --dir_in="${input_dir}/${id}" \
                                  --dir_out="${output_dir}/$(printf "%03d" $id)" \
                                  --save_bilinear_vertices True
    done
}


# Loop through the number of CPU cores
for ((i=0; i<$NUM_CORES; i++)); do
    start_index=$(($i * $SUBJECTS_PER_PROCESS + 1))
    end_index=$(($(($i + 1)) * $SUBJECTS_PER_PROCESS))
    if [ $end_index -gt $TOTAL_SUBJECTS ]; then
        end_index=$TOTAL_SUBJECTS
    fi
    process_subjects $start_index $end_index &
done

wait