export INPUT_IMG=./demo/input/stylegan2.png
export TARGET_EXPRESSION_IMG=./demo/exp/smile.png

rm -r ./third_party/MICA/demo/
rm -r ./third_party/metrical-tracker/input
rm -r ./third_party/metrical-tracker/output

# predict shape parameters with MICA
mkdir -p ./third_party/MICA/demo/input/
cp $INPUT_IMG ./third_party/MICA/demo/input/input.png
cp $TARGET_EXPRESSION_IMG ./third_party/MICA/demo/input/exp.png
cd ./third_party/MICA/
python demo.py
cd ../..

# fit flame mesh with MICA
mkdir -p ./third_party/metrical-tracker/input/input/source
cp $TARGET_EXPRESSION_IMG ./third_party/metrical-tracker/input/input/source/00001.png
cd ./third_party/metrical-tracker/
python tracker.py --cfg ./configs/actors/config.yml
cd ../..

# inference with Morphable Diffusion
python generate_face.py --input_img $INPUT_IMG \
                   --mesh ./third_party/metrical-tracker/output/config/mesh/00001.ply \
                   --ckpt './ckpt/facescape_flame.ckpt' \
                   --output_dir './output/' \
                   --cfg_scale 2.0 \
                   --batch_view_num 8 \
                   --seed 6033 \
                   --sample_steps 50 \