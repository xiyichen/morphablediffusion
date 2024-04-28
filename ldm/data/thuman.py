import pytorch_lightning as pl
import numpy as np
import torch
import PIL
import os
from skimage.io import imread
import webdataset as wds
import PIL.Image as Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from ldm.base_utils import read_pickle
import torchvision.transforms as transforms
import torchvision
from einops import rearrange

from scipy.spatial.transform import Rotation as R
import random, trimesh

class THumanData(Dataset):
    def __init__(self, data_dir, smplx_dir, uids, image_size=256):
        self.default_image_size = 256
        self.image_size = image_size
        self.data_dir = Path(data_dir)
        self.smplx_dir = Path(smplx_dir)

        self.uids = uids
        print('============= length of dataset %d =============' % len(self.uids))

        image_transforms = []
        image_transforms.extend([transforms.ToTensor(), transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)
        self.num_images = 16

    def __len__(self):
        return len(self.uids)

    def load_im(self, path):
        img = imread(path)
        img = img.astype(np.float32) / 255.0
        mask = img[:,:,3:]
        img[:,:,:3] = img[:,:,:3] * mask + 1 - mask # white background
        img_np = np.uint8(img[:, :, :3] * 255.)
        img_pil = Image.fromarray(img_np)
        return img_pil, mask, img_np

    def process_im(self, im):
        im = im.convert("RGB")
        im = im.resize((self.image_size, self.image_size), resample=PIL.Image.BICUBIC)
        return self.image_transforms(im)

    def load_index(self, data_dir, view_id):
        img, _, img_np = self.load_im(os.path.join(data_dir, f'{str(view_id).zfill(3)}.png'))
        preprocessed_img = self.process_im(img)
        return preprocessed_img, img_np
    
    def get_data_for_index(self, idx):
        uid = str(self.uids[idx]).zfill(4)
        target_views = list(range(self.num_images))
        random.shuffle(target_views)
        
        target_images = []
        target_elevations = []
        target_azimuths = []
        target_Ks = []
        target_RTs = []
        K, _, _, _, poses = read_pickle('./assets/thuman_meta.pkl')
        for target_view in target_views:
            img, _ = self.load_index(os.path.join(self.data_dir, 'target', uid), target_view)
            target_images.append(img)
            target_elevations.append(0)
            target_azimuths.append(0)
            RT = np.array(poses[target_view])
            target_Ks.append(K)
            target_RTs.append(RT)
        target_Ks = torch.tensor(np.array(target_Ks)).float()
        target_RTs = torch.tensor(np.array(target_RTs)).float()
        
        target_images = torch.stack(target_images, 0)
        
        target_elevations = torch.from_numpy(np.array(target_elevations).astype(np.float32))
        target_azimuths = torch.from_numpy(np.array(target_azimuths).astype(np.float32))
        
        input_view = random.randint(0, 15)
        input_img, _ = self.load_index(os.path.join(self.data_dir, 'input', uid), input_view)
        input_elevation = torch.tensor([0]).float()
        input_azimuth = torch.tensor([0]).float()
        input_K, _, _, _, input_poses = read_pickle(os.path.join(self.data_dir, 'input', uid, 'meta.pkl'))
        input_RT = np.array(input_poses[input_view])
        if int(uid) >= 526:
            rot_blender = np.eye(3)
        else:
            rot_blender = np.array([[1,0,0],[0,0,-1],[0,1,0]])
        
        mesh = trimesh.load(os.path.join(self.smplx_dir, uid, 'mesh_smplx.obj'), process=False)
        smpl_verts_gt = mesh.vertices
        smpl_verts_gt = (rot_blender@smpl_verts_gt.transpose(1,0)).transpose(1,0)
        
        normalization = np.load(os.path.join(self.data_dir, 'normalization', f'{uid}.npy'), allow_pickle=True)
        normalization = np.array(normalization).astype(np.float32)
        
        vertices = torch.from_numpy((smpl_verts_gt * normalization[0] + normalization[1:])).float()
        
        min_xyz = torch.min(vertices, axis=0).values
        max_xyz = torch.max(vertices, axis=0).values
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        dhw = vertices[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = torch.tensor([0.005, 0.005, 0.005])
        coord = torch.round((dhw - min_dhw) / voxel_size).int()
        out_sh = torch.ceil((max_dhw - min_dhw) / voxel_size).int()
        x = 4
        out_sh = (out_sh | (x - 1)) + 1
        
        return {"target_image": target_images, "input_image": input_img, "input_elevation": input_elevation,
                "input_azimuth": input_azimuth, "input_K": input_K, "input_RT": input_RT, "target_elevation": target_elevations,
                "target_azimuth": target_azimuths, "target_K": target_Ks, "target_RT": target_RTs, "vertices": vertices,
                "out_sh": out_sh, "coord": coord, "bounds": bounds}

    def __getitem__(self, index):
        data = self.get_data_for_index(index)
        return data

class THumanDataset(pl.LightningDataModule):
    def __init__(self, data_dir=None, smplx_dir=None, batch_size=1, uid_set_pkl=None, image_size=256, num_workers=4, seed=0, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.smplx_dir = smplx_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uid_set_pkl = uid_set_pkl
        self.seed = seed
        self.additional_args = kwargs
        self.image_size = image_size

    def setup(self, stage):
        if stage in ['fit']:
            self.train_dataset = THumanData(self.data_dir, self.smplx_dir, uids=list(range(2201)), image_size=256)
            self.val_dataset = THumanData(self.data_dir, self.smplx_dir, uids=list(range(2201,2445)))
        else:
            raise NotImplementedError
        
    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, seed=self.seed)
        return wds.WebLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        loader = wds.WebLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return loader