# Data Preparation
Before installation, you need to create an account on the [FLAME website](https://flame.is.tue.mpg.de/) and prepare your login and password beforehand. You will be asked to provide them in the download script.

## Download assets and pre-trained models for inference

```bash
bash download_data.sh
```


## Prepare training data (Optional)
### Download training assets

```bash
cd ckpt
gdown 1Wi5GmNEDLmLYvO-jCHzT2OtCAercHOic # Pretrained SyncDreamer model
cd ../assets/
gdown 185t69roYEuhVnRq5D33KMI7F7-pu2oG_ # fitted FLAME vertices for FaceScape
gdown 16FdCGEvC-t8EoMZFbhk6HGllhTEVlpiZ # cleaned up SMPL-X vertices for THuman 2.1
unzip thuman_smplx.zip
rm thuman_smplx.zip
cd ..
```

### Preprocess FaceScape data
Our preprocess script for FaceScape is based on [DINER](). Due to the slow color calibration, it is highly advised to parallelize the process.
#### 1. Request access for the Facescape dataset via https://facescape.nju.edu.cn/Page_Download/
#### 2. Download the Multi-view data (images, camera parameters, and reconstructed 3D shapes)
#### 3. Extract the downloaded files all into `FACESCAPE_RAW`. After extraction, the directory structure should look like this:
    ```
    - FACESCAPE_RAW (dataset root)
    |- 1
    |  |- 1_neutral
    |  |  |- 0.jpg
    |  |  |- 1.jpg
    |  |  |- ...
    |  |  |- 54.jpg
    |  |  |- params.json
    |  |- 1_neutral.ply
    |  |- 2_smile
    |  |- 2_smile.ply
    |  |- ...
    |  |- dpmap
    |  |  |- 1_neutral.png
    |  |  |- 2_smile.png
    |  |  |- ...
    |  |  |- ...
    |  |- models_reg
    |  |  |- 1_neutral.obj
    |  |  |- 1_neutral.jpg
    |  |  |- 1_neutral.mtl
    |  |  |- 2_smile.obj
    |  |  |- 2_smile.jpg
    |  |  |- 2_smile.mtl
    |  |  |- ...
    |- 2
    |- ...
    ```
#### 4. Run the preprocessing script:
```bash
cd preprocessing/facescape
```

For Slurm clusters with OpenMPI support:
```bash
sbatch process_all_mpi.sh
```

For single machine:
```bash
bash process_all.sh
```

### Preprocess THuman data
#### 1. Request access for the THuman 2.1 dataset via https://github.com/ytrock/THuman2.0-Dataset/blob/main/THUman2.0_Agreement.pdf
#### 2. Extract the downloaded files all into `thuman_2.1`. After extraction, the directory structure should look like this:
    ```
    - thuman_2.1 (dataset root)
    |- 0000
    |  |- 0000.obj
    |  |- material0.jpeg
    |  |- material0.mtl
    |- 0001
    |  |- 0001.obj
    |  |- material0.jpeg
    |  |- material0.mtl
    |- 0002
    |- ...
    ```

#### 3. Run the preprocessing script (requires blender):

Extract SMPL-X scale and translations:
```bash
cd preprocessing/facescape
python get_smplx_scale.py --smplx_dir ../../assets/thuman_smplx --output_dir OUTPUT_DIR
```

For Slurm clusters with OpenMPI support:
```bash
sbatch render_batch_mpi.sh
```

For single machine:
```bash
python render_batch.py --input_dir ../../assets/thuman_smplx --output_dir OUTPUT_DIR
```