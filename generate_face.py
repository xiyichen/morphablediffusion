import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imsave
import trimesh
import cv2, os, json

from ldm.models.diffusion.morphable_diffusion import SyncMultiviewDiffusion, SyncDDIMSampler
from ldm.util import instantiate_from_config
import PIL
import PIL.Image as Image
import torchvision.transforms as transforms
import torchvision, pickle
from einops import rearrange
from pytorch3d.transforms import so3_exponential_map
from scipy.spatial.transform import Rotation as Rot

image_transforms = []
image_transforms.extend([transforms.ToTensor(), transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
image_transforms = torchvision.transforms.Compose(image_transforms)

def generate_camera_trajectory(num_cameras=16):
    # Constants
    radius = 4.5
    x_angle = -180
    z_angle = 0
    
    angles = np.linspace(-90, 90, num_cameras)
    camera_positions = []
    rotation_matrices = []
    
    for y_angle in angles:
        y_angle_rad = np.radians(y_angle)
        
        x_pos = radius * np.sin(y_angle_rad)
        z_pos = radius * np.cos(y_angle_rad)
        camera_positions.append((x_pos, 0, z_pos))
        
        rotation_matrix = (x_angle, y_angle, z_angle)
        rotation_matrices.append(rotation_matrix)
    
    return camera_positions, rotation_matrices

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
    parser.add_argument('--exp_img',type=str, required=True)
    parser.add_argument('--mesh', type=str, required=True)
    parser.add_argument('--cfg',type=str, default='configs/facescape.yaml')
    parser.add_argument('--ckpt',type=str, default='ckpt/facescape_flame.ckpt')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--batch_view_num', type=int, default=8)
    parser.add_argument('--seed', type=int, default=6033)

    parser.add_argument('--sampler', type=str, default='ddim')
    parser.add_argument('--sample_steps', type=int, default=50)
    parser.add_argument('--camera_trajectory', type=str, default='virtual', choices=['real', 'virtual'])
    parser.add_argument('--prepare_neus2_data', action='store_true')
    flags = parser.parse_args()
    
    img_name = flags.input_img.split('/')[-1].split('.')[0]
    exp_name = flags.exp_img.split('/')[-1].split('.')[0]

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
    
    if flags.camera_trajectory == 'real':
        with open('./assets/facescape_test_traj.pkl', 'rb') as f:
            camera_dict = pickle.load(f)
    elif flags.camera_trajectory == 'virtual':
        cameras = generate_camera_trajectory(16)
    else:
        raise NotImplementedError

    if flags.prepare_neus2_data:
        neus2_data_root = os.path.join(flags.output_dir, 'neus2_data', f'{img_name}_{exp_name}')
        os.makedirs(os.path.join(neus2_data_root, 'images'), exist_ok=True)
        d = {}
        d['w'] = 256
        d['h'] = 256
        d['aabb_scale'] = 1.0
        d['scale'] = 1.0
        d['offset'] = [0.5,0.5,0.5]
        d['frames'] = []
    
    for idx in range(16):
        target_images.append(input_img)
        target_elevations.append(0)
        target_azimuths.append(0)
        
        K = np.eye(4)
        if flags.camera_trajectory == 'real':
            K[:3,:3] = np.array(camera_dict['intrinsics'][idx])
            RT = np.array(camera_dict['extrinsics'][idx])
        else:
            K[:3,:3] = np.array([[1545.23757707405, 0.0, 128.0], [0.0, 1545.23757707405, 128.0], [0.0, 0.0, 1.0]])
            position = np.array(cameras[0][idx])
            rotation = np.array(cameras[1][idx])
            R = Rot.from_euler('xyz', rotation, True).as_matrix()
            t = -R@position.reshape(3,1)
            RT = np.zeros((3,4))
            RT[:3,:3] = R
            RT[:3,3] = t.reshape(3,)
        
        if flags.prepare_neus2_data:
            E = np.eye(4)
            E[:3,:4] = RT
            c2w = np.linalg.inv(E)
            c2w[:,1] *= -1
            c2w[:,2] *= -1
            d_curr = {}
            d_curr['file_path'] = f'images/{str(idx).zfill(2)}.png'
            d_curr['transform_matrix'] = c2w.tolist()
            d_curr['intrinsic_matrix'] = K[:3,:3].tolist()
            d['frames'].append(d_curr)
        
        target_Ks.append(K)
        target_RTs.append(RT)
    
    if flags.prepare_neus2_data:
        with open(os.path.join(neus2_data_root, f'transform.json'), 'w') as f:
            json.dump(d, f, indent=4)
    
    target_Ks = torch.tensor(target_Ks).float()
    target_RTs = torch.tensor(np.array(target_RTs)).float()

    target_images = torch.stack(target_images, 0)
    target_elevations = torch.from_numpy(np.array(target_elevations).astype(np.float32))
    target_azimuths = torch.from_numpy(np.array(target_azimuths).astype(np.float32))
    input_elevation = torch.tensor([0]).float()
    input_azimuth = torch.tensor([0]).float()

    verts = trimesh.load(flags.mesh, process=False).vertices
    face_vertices = torch.from_numpy(verts).float()
    # hard-coded scale and pose to align the MICA-optimized FLAME meshes with the fitted ones for FaceScape used for training
    face_vertices *= 1.087
    pose = torch.tensor([1.6811e+00, -2.6845e-02, -2.8883e-02,  8.5418e-04, -3.4041e-03, 1.0564e-02]).reshape(1,-1)
    R = so3_exponential_map(pose[:,:3])[0]
    T = pose[0,3:]
    face_vertices = (R@face_vertices.T).T + T.reshape(-1, 3)
    face_vertices *= 2.5
    face_vertices = (torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., -1., 0]]).float()@face_vertices.T).T

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
    output_fn = Path(flags.output_dir)/ f'{img_name}_{exp_name}.png'
    n_views = np.concatenate([x_sample[:,ni] for ni in range(N)], 2)
    batch_output = np.concatenate(n_views, 0)
    imsave(output_fn, batch_output)
    
    if flags.prepare_neus2_data:
        for idx in range(16):
            img = batch_output[:, idx*256:(idx+1)*256, :]
            alpha_channel = (~(np.all(img > 240, axis=-1))).astype(np.int8)*255
            img_bgra = np.zeros((256,256,4))
            img_bgra[:,:,:3] = img[:,:,::-1]
            img_bgra[:,:,-1] = alpha_channel
            cv2.imwrite(os.path.join(neus2_data_root, f'images/{str(idx).zfill(2)}.png'), img_bgra)
    
if __name__=="__main__":
    main()