
# Evaluation
## FaceScape

#### 1. Download checkpoints trained with bilinear topology and dlib models (for re-id evaluation):
```bash
cd ckpt
gdown 1Zey3u9B-NQU5ltEsyCDJShdwz9RInfE8 # Novel facial expression synthesis model
gdown 1o0qbl-nGUn1_vfMrV31re7do6LFV4WN8 # Novel view synthesis model
cd ../assets/
mkdir dlib
cd dlib
wget https://github.com/justadudewhohacks/face-recognition.js-models/raw/master/models/shape_predictor_5_face_landmarks.dat
wget https://github.com/justadudewhohacks/face-recognition.js-models/raw/master/models/dlib_face_recognition_resnet_model_v1.dat
cd ../..
conda activate morphable_diffusion
pip install dlib==19.21.0
```

#### 2. Generate all views of FaceScape test subjects. $DATA_DIR is the path of the preprocessed FaceScape data and $MODE is either "nes" (novel facial expression synthesis) or "nvs" (novel view synthesis).
```bash
python eval/get_input_target_views_facescape.py --data_dir $DATA_DIR
python eval/generate_all_facescape.py --data_dir $DATA_DIR --ckpt ./ckpt/facescape_bilinear_novel_exp.ckpt --mode $MODE --output_dir ./eval/facescape_bilinear_nes_output # for novel expression synthesis
python eval/generate_all_facescape.py --data_dir $DATA_DIR --ckpt ./ckpt/facescape_bilinear_novel_view.ckpt --mode $MODE --output_dir ./eval/facescape_bilinear_nvs_output # for novel view synthesis
```

#### 3. Predict facial keypoints for both ground-truth and generated views:
```bash
python eval/predict_keypoints.sh -d $DATA_DIR -m $MODE
```

#### 4. Evaluate 2D metrics:
```bash
python eval/eval_2d_facescape.py --data_dir $DATA_DIR --mode $MODE
```