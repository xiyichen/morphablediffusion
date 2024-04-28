import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imsave
import trimesh
import cv2

from ldm.models.diffusion.morphable_diffusion import SyncMultiviewDiffusion, SyncDDIMSampler
from ldm.util import instantiate_from_config
import PIL
import PIL.Image as Image
import torchvision.transforms as transforms
import torchvision, pickle
from einops import rearrange

image_transforms = []
image_transforms.extend([transforms.ToTensor(), transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
image_transforms = torchvision.transforms.Compose(image_transforms)

class BackgroundRemoval:
    def __init__(self, device='cuda'):
        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)
        image = self.interface([image])[0]
        image = np.array(image)
        return image

def load_model(cfg,ckpt,strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt,map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=False)
    model = model.cuda().eval()
    return model

def process_im(img):
    img = img.astype(np.float32) / 255.0
    mask = img[:,:,3:]
    img[:,:,:3] = img[:,:,:3] * mask + 1 - mask # white background
    img_np = np.uint8(img[:, :, :3] * 255.)
    im = Image.fromarray(img_np)
    im = im.convert("RGB")
    im = im.resize((256, 256), resample=PIL.Image.BICUBIC)
    return image_transforms(im)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img',type=str, required=True)
    parser.add_argument('--mesh', type=str, required=True)
    parser.add_argument('--cfg',type=str, default='configs/facescape.yaml')
    parser.add_argument('--ckpt',type=str, default='ckpt/facescape_flame.ckpt')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--batch_view_num', type=int, default=8)
    parser.add_argument('--seed', type=int, default=6033)

    parser.add_argument('--sampler', type=str, default='ddim')
    parser.add_argument('--sample_steps', type=int, default=50)
    flags = parser.parse_args()

    torch.random.manual_seed(flags.seed)

    target_images = []
    target_elevations = []
    target_azimuths = []
    target_Ks = []
    target_RTs = []

    mask_predictor = BackgroundRemoval()
    image = cv2.imread(flags.input_img, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgba = mask_predictor(image)
    input_img = process_im(rgba)
    
    model = load_model(flags.cfg, flags.ckpt, strict=True)
    assert isinstance(model, SyncMultiviewDiffusion)
    Path(f'{flags.output_dir}').mkdir(exist_ok=True, parents=True)

    if flags.sampler=='ddim':
        sampler = SyncDDIMSampler(model, flags.sample_steps)
    else:
        raise NotImplementedError
    
    with open('./assets/facescape_test_traj.pkl', 'rb') as handle:
        camera_dict = pickle.load(handle)
    for idx in range(16):
        target_images.append(input_img)
        target_elevations.append(0)
        target_azimuths.append(0)
        K = np.eye(4)
        K[:3,:3] = np.array(camera_dict['intrinsics'][idx])
        RT = np.array(camera_dict['extrinsics'][idx])
        target_Ks.append(K)
        target_RTs.append(RT)
    target_Ks = torch.tensor(target_Ks).float()
    target_RTs = torch.tensor(np.array(target_RTs)).float()

    target_images = torch.stack(target_images, 0)
    target_elevations = torch.from_numpy(np.array(target_elevations).astype(np.float32))
    target_azimuths = torch.from_numpy(np.array(target_azimuths).astype(np.float32))
    input_elevation = torch.tensor([0]).float()
    input_azimuth = torch.tensor([0]).float()

    verts = trimesh.load(flags.mesh, process=False).vertices
    face_vertices = torch.from_numpy(verts).float()
    face_vertices *= 2.5

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
    data = {}
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

    output_fn = Path(flags.output_dir)/ 'out.png'
    n_views = np.concatenate([x_sample[:,ni] for ni in range(N)], 2)
    batch_output = np.concatenate(n_views, 0)
    imsave(output_fn, batch_output)
    
if __name__=="__main__":
    main()