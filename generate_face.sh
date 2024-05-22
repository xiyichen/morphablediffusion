#!/bin/bash
set -e

export INPUT_IMG=./demo/face/input/stylegan1.png
export TARGET_EXPRESSION_IMG=./demo/face/exp/kiss.png

rm -rf ./third_party/MICA/demo/
rm -rf ./third_party/metrical-tracker/input
rm -rf ./third_party/metrical-tracker/output

# Predict shape parameters with MICA
mkdir -p ./third_party/MICA/demo/input/
cp "$INPUT_IMG" ./third_party/MICA/demo/input/input.png
cp "$TARGET_EXPRESSION_IMG" ./third_party/MICA/demo/input/exp.png
cd ./third_party/MICA/
python demo.py
cd ../../

# Fit flame mesh with MICA
mkdir -p ./third_party/metrical-tracker/input/input/source
cp "$TARGET_EXPRESSION_IMG" ./third_party/metrical-tracker/input/input/source/00001.png
cd ./third_party/metrical-tracker/
python tracker.py --cfg ./configs/actors/config.yml
cd ../../

# Inference with Morphable Diffusion
python generate_face.py --input_img $INPUT_IMG \
                        --exp_img $TARGET_EXPRESSION_IMG \
                        --mesh ./third_party/metrical-tracker/output/config/mesh/00001.ply \
                        --ckpt './ckpt/facescape_flame_mv_tracking.ckpt' \
                        --output_dir './output/' \
                        --cfg_scale 2.0 \
                        --batch_view_num 8 \
                        --seed 6033 \
                        --sample_steps 50