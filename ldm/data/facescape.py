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
import torchvision.transforms as transforms
import torchvision
from einops import rearrange
from scipy.spatial.transform import Rotation as R
import json
import random
import trimesh

class FaceScapeData(Dataset):
    def __init__(self, data_dir, mesh_topology, subjects, expressions, heldout_expressions, image_size=256, shuffled_expression=True):
        self.default_image_size = 256
        self.image_size = image_size
        self.data_dir = Path(data_dir)
        self.mesh_topology = mesh_topology
        self.uids = []
        self.expressions = expressions
        self.heldout_expressions = heldout_expressions
        self.shuffled_expression = shuffled_expression
        
        for s in subjects:
            for e in expressions:
                self.uids.append(f'{s}/{e}')

        print('============= length of dataset %d =============' % len(self.uids))

        image_transforms = []
        image_transforms.extend([transforms.ToTensor(), transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)
        self.num_images = 16
        self.CAPSTUDIO_2_FACESCAPE = torch.tensor([[ 1.,  0.,  0.], [ 0.,  0.,  1.], [-0., -1., -0.]]).float()
        self.FACESCAPE_2_CAPSTUDIO = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0]]).float()

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
        img, _, img_np = self.load_im(os.path.join(data_dir, f'view_{view_id.zfill(5)}/rgba_colorcalib.png'))
        preprocessed_img = self.process_im(img)
        return preprocessed_img, img_np
    
    def get_input_view(self, subject_id, expression_id):
        if self.shuffled_expression:
            possible_exps = list(range(1, 21))
            for exp in self.heldout_expressions:
                possible_exps.remove(int(exp))
            if int(expression_id) in possible_exps:
                possible_exps.remove(int(expression_id))
            exp_id = random.sample(possible_exps, 1)[0]
        else:
            exp_id = int(expression_id)
        data_dir = os.path.join(self.data_dir, f'{subject_id}/{str(exp_id).zfill(2)}')
        with open(os.path.join(data_dir, 'cameras.json'), 'r') as f:
            camera_dict = json.load(f)
        valid_views = []
        for view in camera_dict.keys():
            if os.path.isfile(os.path.join(data_dir, f'view_{str(view).zfill(5)}', 'rgba_colorcalib.png')):
                RT = np.array(camera_dict[view]['extrinsics'])
                if abs(Rot.from_matrix(RT[:3,:3]).as_euler('xyz', True)[-1]) > 90:
                    continue
                valid_views.append(view)
        input_candidates = []
        for valid_view in valid_views:
            if (abs(camera_dict[valid_view]['angles']['azimuth']) <= 40):
                input_candidates.append(valid_view)
        input_idx = random.sample(input_candidates, 1)[0]
        input_elevation = torch.tensor([0]).float()
        input_azimuth = torch.tensor([0]).float()
        input_K = torch.tensor(camera_dict[input_idx]['intrinsics']).float()
        input_RT = torch.tensor(camera_dict[input_idx]['extrinsics']).float()
        input_RT[:3,:3] = input_RT[:3,:3]@self.FACESCAPE_2_CAPSTUDIO.numpy()
        input_RT[:3,3] *= 2.5
        input_img, _ = self.load_index(data_dir, input_idx)
        return input_img, input_elevation, input_azimuth, input_K, input_RT

    def get_data_for_index(self, index):
        idx = index
        while True:
            try:
                data_dir = os.path.join(self.data_dir, self.uids[idx])
                with open(os.path.join(data_dir, 'cameras.json'), 'r') as f:
                    camera_dict = json.load(f)
                subject_id, expression_id = self.uids[idx].split('/')
                
                valid_views = []
                for view in camera_dict.keys():
                    RT = np.array(camera_dict[view]['extrinsics'])
                    # exclude some upside down camera views
                    if abs(Rot.from_matrix(RT[:3,:3]).as_euler('xyz', True)[-1]) > 90:
                        continue
                    if os.path.isfile(os.path.join(data_dir, f'view_{str(view).zfill(5)}', 'rgba_colorcalib.png')):
                        valid_views.append(view)
                target_view_candidates = []
                for valid_view in valid_views:
                    if abs(camera_dict[valid_view]['angles']['azimuth']) <= 90:
                        target_view_candidates.append(valid_view)
                target_views = random.sample(target_view_candidates, self.num_images)
                
                input_img, input_elevation, input_azimuth, input_K, input_RT = self.get_input_view(subject_id, expression_id)
                
                if self.mesh_topology == 'bilinear':
                    face_vertices = 2.5 * torch.from_numpy(np.loadtxt(os.path.join(data_dir, 'face_vertices.npy'))).float()
                    face_vertices = (self.CAPSTUDIO_2_FACESCAPE@face_vertices.T).T
                elif self.mesh_topology == 'flame':
                    face_vertices = 2.5 * torch.from_numpy(trimesh.load(f'./assets/facescape_flame_tracking/{subject_id}/{expression_id}/mesh.obj', process=False).vertices).float()
                    face_vertices = (self.CAPSTUDIO_2_FACESCAPE@face_vertices.T).T
                else:
                    raise NotImplementedError(f"Mesh topology {self.mesh_topology} not supported!")
                
                break
            except Exception as e:
                print(f'ran into exception: {e}')
                idx = random.randint(0, len(self.uids)-1)
                continue
        
        target_images = []
        target_elevations = []
        target_azimuths = []
        target_Ks = []
        target_RTs = []
        for target_view in target_views:
            img, _ = self.load_index(data_dir, target_view)
            target_images.append(img)
            target_elevations.append(0)
            target_azimuths.append(0)
            K = np.eye(4)
            K[:3,:3] = np.array(camera_dict[target_view]['intrinsics'])
            RT = np.array(camera_dict[target_view]['extrinsics'])
            RT[:3,3] *= 2.5
            RT[:3,:3] = RT[:3,:3]@self.FACESCAPE_2_CAPSTUDIO.numpy()
            
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
        
        return {"target_image": target_images, "input_image": input_img, "input_elevation": input_elevation,
                "input_azimuth": input_azimuth, "input_K": input_K, "input_RT": input_RT, "target_elevation": target_elevations,
                "target_azimuth": target_azimuths, "target_K": target_Ks, "target_RT": target_RTs, "vertices": face_vertices,
                "out_sh": out_sh, "coord": coord, "bounds": bounds}

    def __getitem__(self, index):
        data = self.get_data_for_index(index)
        return data

class FaceScapeDataset(pl.LightningDataModule):
    def __init__(self, mesh_topology='bilinear', shuffled_expression=True, data_dir=None, validation_dir=None, batch_size=1, uid_set_pkl=None, image_size=256, num_workers=4, seed=0, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.validation_dir = validation_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uid_set_pkl = uid_set_pkl
        self.seed = seed
        self.additional_args = kwargs
        self.image_size = image_size
        self.mesh_topology = mesh_topology
        self.shuffled_expression = shuffled_expression

    def setup(self, stage):
        if stage in ['fit']:
            heldout_expressions = ['06']
            train_subjects = [str(i).zfill(3) for i in list(range(1, 326))]
            for i in ['122', '212']:
                train_subjects.remove(i)
            test_subjects = ['122', '212'] + [str(i) for i in range(326, 360)]
            train_expressions = [str(i).zfill(2) for i in range(1,21)]
            for exp in heldout_expressions:
                train_expressions.remove(exp)
            test_expressions = heldout_expressions
            self.train_dataset = FaceScapeData(self.data_dir, mesh_topology=self.mesh_topology, subjects=train_subjects, expressions=train_expressions, heldout_expressions=heldout_expressions, shuffled_expression=self.shuffled_expression)
            self.val_dataset = FaceScapeData(self.data_dir, mesh_topology=self.mesh_topology, subjects=test_subjects, expressions=test_expressions, heldout_expressions=heldout_expressions, shuffled_expression=self.shuffled_expression)
        else:
            raise NotImplementedError
        
    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, seed=self.seed)
        return wds.WebLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)


    def val_dataloader(self):
        loader = wds.WebLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return loader