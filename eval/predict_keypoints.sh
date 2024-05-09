#!/bin/bash

# Default value for the DATA_DIR variable
DATA_DIR=""
MODE=""

# Usage function to display help
usage() {
    echo "Usage: $0 -d <data_dir> -m <mode>"
    echo "  -d  Path to the preprocessed FaceScape data directory (required)"
    echo "  -m  Mode of operation (required): must be 'nes' (for novel facial expression synthesis) or 'nvs' (for novel view synthesis)"
    exit 1
}

# Parse command-line arguments
while getopts d:m: flag; do
    case "${flag}" in
        d) DATA_DIR=${OPTARG};;
        m) MODE=${OPTARG};;
        *) usage;;  # For any unrecognized options
    esac
done

# Validate that both required options have been provided
if [ -z "$DATA_DIR" ] || [ -z "$MODE" ]; then
    echo "Error: Both -d and -m are required."
    usage
fi

# Validate the mode
if [[ "$MODE" != "nes" && "$MODE" != "nvs" ]]; then
    echo "Error: The mode must be either 'nes' or 'nvs'."
    usage
fi

# Print the values for verification
echo "Data Directory: $DATA_DIR"
echo "Mode: $MODE"

echo "Predicting keypoints from ground truth views"
python eval/predict_keypoints.py \
    ./third_party/mmpose/demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/yolo-x_8xb8-300e_coco-face_13274d7c.pth \
    ./third_party/mmpose/configs/face_2d_keypoint/topdown_heatmap/coco_wholebody_face/td-hm_hrnetv2-w18_dark-8xb32-60e_coco-wholebody-face-256x256.py \
    https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_coco_wholebody_face_256x256_dark-3d9a334e_20210909.pth \
    --data_dir $DATA_DIR --gt --mode $MODE --save_projections

echo "Predicting keypoints from generated views"
python eval/predict_keypoints.py \
    ./third_party/mmpose/demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/yolo-x_8xb8-300e_coco-face_13274d7c.pth \
    ./third_party/mmpose/configs/face_2d_keypoint/topdown_heatmap/coco_wholebody_face/td-hm_hrnetv2-w18_dark-8xb32-60e_coco-wholebody-face-256x256.py \
    https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_coco_wholebody_face_256x256_dark-3d9a334e_20210909.pth \
    --data_dir $DATA_DIR --pred_dir ./eval/facescape_bilinear_${MODE}_output --mode $MODE --save_projections