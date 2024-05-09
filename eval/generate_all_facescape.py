import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imsave
import trimesh
import cv2
import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from ldm.models.diffusion.morphable_diffusion import SyncMultiviewDiffusion, SyncDDIMSampler
from ldm.util import instantiate_from_config
import PIL
import PIL.Image as Image
import torchvision.transforms as transforms
import torchvision, pickle
from einops import rearrange
import glob
import json
import random
import math
from skimage.io import imread

image_transforms = []
image_transforms.extend([transforms.ToTensor(), transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
image_transforms = torchvision.transforms.Compose(image_transforms)

def load_model(cfg,ckpt,strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt,map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=False)
    model = model.cuda().eval()
    return model

def load_im(path):
    img = imread(path)
    img = img.astype(np.float32) / 255.0
    mask = img[:,:,3:]
    img[:,:,:3] = img[:,:,:3] * mask + 1 - mask # white background
    img_np = np.uint8(img[:, :, :3] * 255.)
    img_pil = Image.fromarray(img_np)
    return img_pil, mask, img_np

def process_im(im):
    im = im.convert("RGB")
    return image_transforms(im)

def load_index(data_dir, view_id):
    img, _, img_np = load_im(os.path.join(data_dir, f'view_{view_id.zfill(5)}/rgba_colorcalib.png'))
    preprocessed_img = process_im(img)
    return preprocessed_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['nvs', 'nes'])
    parser.add_argument('--cfg', type=str, default='./configs/facescape.yaml')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./eval/facescape_bilinear_nes_output')
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--batch_view_num', type=int, default=8)
    parser.add_argument('--seed', type=int, default=6033)

    parser.add_argument('--sampler', type=str, default='ddim')
    parser.add_argument('--sample_steps', type=int, default=50)
    flags = parser.parse_args()

    torch.random.manual_seed(flags.seed)
    
    test_subjects = [str(i) for i in [122, 212] + list(range(326, 360))]
    if flags.mode == 'nes':
        test_exps = [6]
    else:
        test_exps = list(range(1, 21))
    test_exps = [str(i).zfill(2) for i in test_exps]

    model = load_model(flags.cfg, flags.ckpt, strict=True)
    assert isinstance(model, SyncMultiviewDiffusion)
    Path(f'{flags.output_dir}').mkdir(exist_ok=True, parents=True)
    
    if flags.sampler=='ddim':
        sampler = SyncDDIMSampler(model, flags.sample_steps)
    else:
        raise NotImplementedError
    
    with open(os.path.join('./eval/facescape_input_target_views.json')) as f:
        test_metadata = json.load(f)
        
    CAPSTUDIO_2_FACESCAPE = torch.tensor([[ 1.,  0.,  0.], [ 0.,  0.,  1.], [-0., -1., -0.]]).float()
    FACESCAPE_2_CAPSTUDIO = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0]]).float()
    
    for subject_id in test_subjects:
        for expression_id in test_exps:
            data_dir = os.path.join(flags.data_dir, f'{subject_id}/{expression_id}')
            views = glob.glob(os.path.join(data_dir, 'view_*'))
            if len(views) == 0:
                continue
            
            with open(os.path.join(data_dir, 'cameras.json'), 'r') as f:
                camera_dict = json.load(f)
            all_target_views = test_metadata[str(subject_id).zfill(3)][str(expression_id).zfill(2)]['target_views']
            # round up to multiples of 16 to generate in batches
            all_target_views = all_target_views + all_target_views[:(math.ceil(len(all_target_views)/16)*16 - len(all_target_views))]
            with open(os.path.join(data_dir, 'cameras.json'), 'r') as f:
                camera_dict = json.load(f)
            if flags.mode == 'nes':
                possible_exps = list(range(1, 21))
                possible_exps.remove(int(expression_id))
                input_exp = str(random.sample(possible_exps, 1)[0]).zfill(2)
            else:
                input_exp = expression_id
            
            input_view = test_metadata[str(subject_id).zfill(3)][str(input_exp).zfill(2)]['input_view']
            
            input_img = load_index(os.path.join(flags.data_dir, f'{subject_id}/{input_exp}'), input_view)
            input_elevation = torch.tensor([0]).float()
            input_azimuth = torch.tensor([0]).float()
            
            face_vertices = 2.5 * torch.from_numpy(np.loadtxt(os.path.join(data_dir, 'face_vertices.npy'))).float()
            face_vertices = (CAPSTUDIO_2_FACESCAPE@face_vertices.T).T
            
            data = {}
            for b in range(len(all_target_views)//16):
                target_views = all_target_views[b*16:b*16+16]
                
                target_images = []
                target_elevations = []
                target_azimuths = []
                target_Ks = []
                target_RTs = []
                
                for target_view in target_views:
                    img = load_index(data_dir, target_view)
                    target_images.append(img)
                    target_elevations.append(0)
                    target_azimuths.append(0)
                    K = np.eye(4)
                    K[:3,:3] = np.array(camera_dict[target_view]['intrinsics'])
                    RT = np.array(camera_dict[target_view]['extrinsics'])
                    RT[:3,3] *= 2.5
                    RT[:3,:3] = RT[:3,:3]@FACESCAPE_2_CAPSTUDIO.numpy()
                    target_Ks.append(K)
                    target_RTs.append(RT)
                
                target_Ks = torch.tensor(np.array(target_Ks)).float()
                target_RTs = torch.tensor(np.array(target_RTs)).float()
                
                target_images = torch.stack(target_images, 0)
                
                target_elevations = torch.from_numpy(np.array(target_elevations).astype(np.float32))
                target_azimuths = torch.from_numpy(np.array(target_azimuths).astype(np.float32))
                
                min_xyz = torch.min(face_vertices, axis=0).values
                max_xyz = torch.max(face_vertices, axis=0).values
                bounds = np.stack([min_xyz, max_xyz], axis=0)
                dhw = face_vertices[:, [2, 1, 0]]
                min_dhw = min_xyz[[2, 1, 0]]
                max_dhw = max_xyz[[2, 1, 0]]
                voxel_size = torch.tensor([0.005, 0.005, 0.005])
                coord = torch.round((dhw - min_dhw) / voxel_size).int()
                out_sh = torch.ceil((max_dhw - min_dhw) / voxel_size).int()
                x = 4
                out_sh = (out_sh | (x - 1)) + 1
                
                data_ = {"target_image": target_images, "input_image": input_img, "input_elevation": input_elevation,
                    "input_azimuth": input_azimuth, "target_elevation": target_elevations,
                    "target_azimuth": target_azimuths, "target_K": target_Ks, "target_RT": target_RTs, "vertices": face_vertices,
                    "out_sh": out_sh, "coord": coord, "bounds": bounds}
                
                for k, v in data_.items():
                    if k not in data:
                        data[k] = []
                    if torch.is_tensor(v):
                        data[k].append(v.unsqueeze(0).cuda())
                    else:
                        data[k].append(torch.from_numpy(v).unsqueeze(0).cuda())
                
            for k in data:
                data[k] = torch.concat(data[k])
            
            
            x_sample = model.sample(sampler, data, flags.cfg_scale, flags.batch_view_num)
            x_sample = torch.concat([data['input_image'].unsqueeze(1).permute(0,1,4,2,3), x_sample], axis=1)

            B, N, _, H, W = x_sample.shape
            x_sample = (torch.clamp(x_sample,max=1.0,min=-1.0) + 1) * 0.5
            x_sample = x_sample.permute(0,1,3,4,2).cpu().numpy() * 255
            x_sample = x_sample.astype(np.uint8)

            output_fn = Path(flags.output_dir)/ f'{subject_id}_{expression_id}.png'
            n_views = np.concatenate([x_sample[:,ni] for ni in range(N)], 2)
            batch_output = np.concatenate(n_views, 0)
            imsave(output_fn, batch_output)
    
if __name__=="__main__":
    main()