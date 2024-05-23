# Morphable Diffusion (CVPR 2024)
Morphable Diffusion: 3D-Consistent Diffusion for Single-image Avatar Creation

![](assets/teaser.png)
![](assets/in_the_wild.png)


## [Project page](https://xiyichen.github.io/morphablediffusion) | [Paper](https://arxiv.org/abs/2401.04728)

### TODOs:
- [x] Release pretrained face model with FLAME topology and inference script.
- [x] Release face model trained with bilinear topology and evaluation code.
- [x] Retrain face model with flame meshes obtained from more accurate FLAME-tracking.
- [x] Allow exporting preprocessed data in NeuS2 format for mesh reconstruction.
- [ ] Release pretrained model, inference script, and evaluation code for the full body model.
- [ ] Update project page and video.

### Instructions
- See [installation](docs/installation.md) to install all the required packages
- See [data preparation](docs/data.md) to set up the download pretrained models, assets and datasets
- See [evaluation](docs/eval.md) to reproduce the qualitative and quantitative results in our paper

### Inference
To run Morphable Diffusion for novel facial expression synthesis, replace `$INPUT_IMG` and `$TARGET_EXPRESSION_IMG` with paths to the input image and target facial expression image and run:
```bash
bash generate_face.sh
```
You could also choose between "virtual" and "real" for the `--camera_trajectory` flag to use a hemispherical virtual camera path or a similar path but with ground truth camera parameters seen during training.
We tested the inference on a single RTX 3090 GPU, but it's also possible to run inference using GPUs with smaller memory (~10GB), e.g., RTX 2080 Ti.

### Mesh Reconstruction
During inference of the face model, we also export the preprocessed data from the generated face images in NeuS2 format, which is stored in `./output_face/neus2_data`. You could then follow the instruction of [NeuS2](https://github.com/19reborn/NeuS2) to reconstruct the mesh. We recommend training NeuS2 using `base.json` as config for 5000 steps.

### Training
We train our models using 2 80GB NVIDIA A100 GPUs with a total batch size of 140.
#### Train the face model:
```bash
bash train_face.sh
```
Please pay attention to the following arguments in `configs/facescape.yaml`:

`data_dir`: path to the preprocessed FaceScape data

`finetune_unet`: Whether to finetune the UNet along with the conditioning module.

`mesh_topology`: The topology of the face mesh used for training (Choose between `flame` and `bilinear`).

`shuffled_expression`: Whether to use different facial expressions for the input and target views. (Discussed in section 4.2 of the paper)

#### Train the full body model:
```bash
bash train_body.sh
```

Please pay attention to the following arguments in `configs/thuman.yaml`:

`data_dir`: path to the preprocessed Thuman 2.1 data.

`smplx_dir`: path to the SMPL-X meshes of the Thuman 2.1 dataset.

`finetune_unet`: Whether to finetune the UNet along with the conditioning module.


During training, we will run validation to output images to `<log_dir>/<images>/val` every 250 steps.

## Acknowledgement

We have extensively borrowed code from the following repositories. Many thanks to the authors for sharing their work.

- [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer)
- [neuralbody](https://github.com/zju3dv/neuralbody)
- [DINER](https://github.com/malteprinzler/diner)
- [MICA](https://github.com/Zielon/MICA)
- [metrical-tracker](https://github.com/Zielon/metrical-tracker.git)

## Citation
If you find this repository useful in your project, please cite the following work. :)
```
@article{chen2024morphable,
         title={Morphable Diffusion: 3D-Consistent Diffusion for Single-image Avatar Creation}, 
         author={Xiyi Chen and Marko Mihajlovic and Shaofei Wang and Sergey Prokudin and Siyu Tang},
         booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
         year={2024}
      }
```
