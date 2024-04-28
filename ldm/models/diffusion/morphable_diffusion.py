from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.io import imsave
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ldm.base_utils import read_pickle, concat_images_list
from ldm.models.diffusion.utils import get_warp_coordinates, create_target_volume, interpolate_features
from ldm.models.diffusion.network import NoisyTargetViewEncoder, SpatialTime3DNet, FrustumTV3DNet, SparseConvNet, SMPLFeatureExtractor
from ldm.modules.diffusionmodules.util import make_ddim_timesteps, timestep_embedding
from ldm.modules.encoders.modules import FrozenCLIPImageEmbedder
from ldm.util import instantiate_from_config
# import open3d as o3d
from spconv.pytorch.core import SparseConvTensor
import pdb
import trimesh, math

def set_lr(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr

def warmup_scheduler(step, warmup_steps):
    if step < warmup_steps:
        return step / warmup_steps
    return 1

def cosine_decay_scheduler(step, decay_steps, total_steps, decay_first=True):
    # decay the last "decay_steps" steps from 1 to 0 using cosine decay
    # if decay_first is True, then the first "decay_steps" steps will be decayed from 1 to 0
    # if decay_first is False, then the last "decay_steps" steps will be decayed from 1 to 0
    if step >= total_steps:
        return 0
    if decay_first:
        if step >= decay_steps:
            return 0
        return (math.cos((step) / decay_steps * math.pi) + 1) / 2
    else:
        if step < total_steps - decay_steps:
            return 1
        return (
            math.cos((step - (total_steps - decay_steps)) / decay_steps * math.pi) + 1
        ) / 2

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def disable_training_module(module: nn.Module):
    module = module.eval()
    module.train = disabled_train
    for para in module.parameters():
        para.requires_grad = False
    return module

def repeat_to_batch(tensor, B, VN):
    t_shape = tensor.shape
    ones = [1 for _ in range(len(t_shape)-1)]
    tensor_new = tensor.view(B,1,*t_shape[1:]).repeat(1,VN,*ones).view(B*VN,*t_shape[1:])
    return tensor_new

class UNetWrapper(nn.Module):
    def __init__(self, diff_model_config, drop_conditions=False, drop_scheme='default', use_zero_123=True):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.drop_conditions = drop_conditions
        self.drop_scheme=drop_scheme
        self.use_zero_123 = use_zero_123

    def drop(self, cond, mask):
        shape = cond.shape
        B = shape[0]
        cond = mask.view(B,*[1 for _ in range(len(shape)-1)]) * cond
        return cond

    def get_trainable_parameters(self):
        return self.diffusion_model.get_trainable_parameters()

    def get_drop_scheme(self, B, device):
        if self.drop_scheme=='default':
            random = torch.rand(B, dtype=torch.float32, device=device)
            drop_clip = (random > 0.15) & (random <= 0.2)
            drop_volume = (random > 0.1) & (random <= 0.15)
            drop_concat = (random > 0.05) & (random <= 0.1)
            drop_all = random <= 0.05
        else:
            raise NotImplementedError
        return drop_clip, drop_volume, drop_concat, drop_all

    def forward(self, x, t, clip_embed, volume_feats, x_concat, is_train=False):
        """

        @param x:             B,4,H,W
        @param t:             B,
        @param clip_embed:    B,M,768
        @param volume_feats:  B,C,D,H,W
        @param x_concat:      B,C,H,W
        @param is_train:
        @return:
        """
        if self.drop_conditions and is_train:
            B = x.shape[0]
            drop_clip, drop_volume, drop_concat, drop_all = self.get_drop_scheme(B, x.device)

            clip_mask = 1.0 - (drop_clip | drop_all).float()
            clip_embed = self.drop(clip_embed, clip_mask)

            volume_mask = 1.0 - (drop_volume | drop_all).float()
            for k, v in volume_feats.items():
                volume_feats[k] = self.drop(v, mask=volume_mask)

            concat_mask = 1.0 - (drop_concat | drop_all).float()
            x_concat = self.drop(x_concat, concat_mask)

        if self.use_zero_123:
            # zero123 does not multiply this when encoding, maybe a bug for zero123
            first_stage_scale_factor = 0.18215
            x_concat_ = x_concat * 1.0
            x_concat_[:, :4] = x_concat_[:, :4] / first_stage_scale_factor
        else:
            x_concat_ = x_concat

        x = torch.cat([x, x_concat_], 1)
        pred = self.diffusion_model(x, t, clip_embed, source_dict=volume_feats)
        return pred

    def predict_with_unconditional_scale(self, x, t, clip_embed, volume_feats, x_concat, unconditional_scale):
        x_ = torch.cat([x] * 2, 0)
        t_ = torch.cat([t] * 2, 0)
        clip_embed_ = torch.cat([clip_embed, torch.zeros_like(clip_embed)], 0)

        v_ = {}
        for k, v in volume_feats.items():
            v_[k] = torch.cat([v, torch.zeros_like(v)], 0)

        x_concat_ = torch.cat([x_concat, torch.zeros_like(x_concat)], 0)
        if self.use_zero_123:
            # zero123 does not multiply this when encoding, maybe a bug for zero123
            first_stage_scale_factor = 0.18215
            x_concat_[:, :4] = x_concat_[:, :4] / first_stage_scale_factor
        x_ = torch.cat([x_, x_concat_], 1)
        s, s_uc = self.diffusion_model(x_, t_, clip_embed_, source_dict=v_).chunk(2)
        s = s_uc + unconditional_scale * (s - s_uc)
        return s

class SpatialVolumeNet(nn.Module):
    def __init__(self, time_dim, view_dim, view_num,
                 input_image_size=256, frustum_volume_depth=48,
                 spatial_volume_size=32, spatial_volume_length=0.5,
                 frustum_volume_length=0.86603, # sqrt(3)/2
                 projection='perspective', use_spatial_volume=False
                 ):
        super().__init__()
        self.target_encoder = NoisyTargetViewEncoder(time_dim, view_dim, output_dim=16)
        self.use_spatial_volume = use_spatial_volume
        if use_spatial_volume:
            self.spatial_volume_feats = SpatialTime3DNet(input_dim=16 * view_num, time_dim=time_dim, dims=(64, 128, 256, 512))
        self.xyzc_net = SparseConvNet()
        self.frustum_volume_feats = FrustumTV3DNet(64, time_dim, view_dim, dims=(64, 128, 256, 512))
        self.smpl_feature_extractor = SMPLFeatureExtractor(filter_channels=[16, 16],
                                                           num_views=16,
                                                           no_residual=False)
        self.projection = projection
        # self.faces = np.array(trimesh.load('/root/1_neutral.obj').faces)

        self.frustum_volume_length = frustum_volume_length
        self.input_image_size = input_image_size
        self.spatial_volume_size = spatial_volume_size
        self.spatial_volume_length = spatial_volume_length

        self.frustum_volume_size = self.input_image_size // 8
        self.frustum_volume_depth = frustum_volume_depth
        self.time_dim = time_dim
        self.view_dim = view_dim
        self.default_origin_depth = 1.5 # our rendered images are 1.5 away from the origin, we assume camera is 1.5 away from the origin

    def construct_spatial_volume(self, x, t_embed, v_embed, batch):
        """
        @param x:            B,N,4,H,W
        @param t_embed:      B,t_dim
        @param v_embed:      B,N,v_dim
        @param target_poses: N,3,4
        @param target_Ks:    N,3,3
        @return:
        """
        B, N, _, H, W = x.shape
        V = self.spatial_volume_size
        
        N_vertices = batch['vertices'][0].shape[0]
        device = x.device
        
        spatial_volume_verts = torch.linspace(-self.spatial_volume_length, self.spatial_volume_length, V, dtype=torch.float32, device=device)
        spatial_volume_verts = torch.stack(torch.meshgrid(spatial_volume_verts, spatial_volume_verts, spatial_volume_verts), -1)
        spatial_volume_verts_ = spatial_volume_verts.reshape(1, V ** 3, 3)[:, :, (2, 1, 0)] # [1,V*V*V,3]
        spatial_volume_verts = spatial_volume_verts_.view(1, V, V, V, 3).permute(0, 4, 1, 2, 3).repeat(B, 1, 1, 1, 1) # [B, 3, V, V, V]
        
        # encode source features
        t_embed_ = t_embed.view(B, 1, self.time_dim).repeat(1, N, 1).view(B, N, self.time_dim)
        v_embed_ = v_embed
        target_Ks = batch['target_K']
        target_poses = batch['target_RT']

        # extract 2D image features
        spatial_volume_feats = []
        # project source features
        for ni in range(0, N):
            pose_source_ = target_poses[:, ni] # B, 3, 4
            K_source_ = target_Ks[:, ni] # B, 3, 3
            x_ = self.target_encoder(x[:, ni], t_embed_[:, ni], v_embed_[:, ni]) # B, C, V, V
            C = x_.shape[1]
            
            coords_source = get_warp_coordinates(spatial_volume_verts, x_.shape[-1], self.input_image_size, K_source_, pose_source_, projection=self.projection).view(B, V, V * V, 2)
            unproj_feats_lowres = F.grid_sample(x_, coords_source, mode='bilinear', padding_mode='zeros', align_corners=True)
            unproj_feats_lowres = unproj_feats_lowres.view(B, C, V, V, V)
            
            spatial_volume_feats.append(unproj_feats_lowres)
        
        spatial_volume_feats = torch.stack(spatial_volume_feats, 1) # B,N,C,V,V,V
        N = spatial_volume_feats.shape[1]
        spatial_volume_feats = spatial_volume_feats.view(B, N*C, V, V, V)
        
        smpl_grid = batch['vertices'].unsqueeze(2).unsqueeze(2) / self.spatial_volume_length # B, N_vertices, 1, 1, 3
        smpl_grids = smpl_grid.unsqueeze(1).repeat(1, N, 1, 1, 1, 1).reshape(B*N, N_vertices, 1, 1, 3)
        smpl_features_views = F.grid_sample(spatial_volume_feats.reshape(B*N, -1, V, V, V), smpl_grids, mode='bilinear', padding_mode='zeros', align_corners=True)[:,:,:,0,0].reshape(B, N, -1, N_vertices) # B, N, 16, N_vertices
        smpl_features = self.smpl_feature_extractor(smpl_features_views) # B, 16, N_vertices
        smpl_features = smpl_features.permute(0, 2, 1) #B, N_vertices, 16
        coords = batch['coord']
        out_shs = batch['out_sh']
        dhw = spatial_volume_verts.permute(0,2,3,4,1).reshape(B, V**3, 3)[:, :, [2, 1, 0]]
        # mesh = trimesh.Trimesh(smpl_verts.detach().cpu().numpy(), self.faces)
        # mesh.export('./0.obj')
        min_dhw = batch['bounds'][:, 0, [2, 1, 0]].unsqueeze(1) # B, 1, 3
        dhw = dhw - min_dhw
        dhw = dhw / torch.tensor([0.005, 0.005, 0.005]).to(device).reshape(1,1,3).repeat(B,1,1)
        # convert the voxel coordinate to [-1, 1]
        dhw = dhw / out_shs.unsqueeze(1) * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]].reshape(B, V, V, V, 3)
        latent_code_volume_feats = []
        for bi in range(B):
            sh = batch['coord'][bi].unsqueeze(0).shape # 1, N_vertices, 3
            idx = [torch.full([sh[1]], i) for i in range(sh[0])]
            idx = torch.cat(idx).to(device)
            coord = coords[bi].view(-1, sh[-1])
            coord = torch.cat([idx[:, None], coord], dim=1).int() # N_vertices, 4
            out_sh = out_shs[bi]
            out_sh = out_sh.int().tolist()
            xyzc = SparseConvTensor(smpl_features[bi], coord, out_sh, 1) # 1, 16, (out_sh)
            feature_volume = self.xyzc_net(xyzc) # 1, 64, (out_sh//4)
            latent_code_volume_feat = F.grid_sample(feature_volume, grid_coords[bi].unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=True) # B, 64, V, V, V
            latent_code_volume_feats.append(latent_code_volume_feat[0])
        volume_feats = torch.stack(latent_code_volume_feats)

        if self.use_spatial_volume:
            spatial_volume_feats = self.spatial_volume_feats(spatial_volume_feats, t_embed)  # b,64,32,32,32
            volume_feats += spatial_volume_feats
        
        return volume_feats

    def construct_view_frustum_volume(self, spatial_volume, t_embed, v_embed, target_indices, batch):
        """
        @param spatial_volume:    B,C,V,V,V
        @param t_embed:           B,t_dim
        @param v_embed:           B,N,v_dim
        @param poses:             N,3,4
        @param Ks:                N,3,3
        @param target_indices:    B,TN
        @return: B*TN,C,H,W
        """
        B, TN = target_indices.shape
        H, W = self.frustum_volume_size, self.frustum_volume_size
        D = self.frustum_volume_depth
        V = self.spatial_volume_size
        
        
        cam_Rs = batch['target_RT'][:,:,:3,:3]
        cam_Ts = batch['target_RT'][:,:,:3,3].unsqueeze(-1)
        cam_positions = torch.einsum('bnij, bnjk -> bnik', -cam_Rs.permute(0,1,3,2), cam_Ts)[:,:,:,0]
        cam_distances = torch.linalg.norm(cam_positions, dim=-1)
        poses = batch['target_RT']
        Ks = batch['target_K']
        poses_ = []
        Ks_ = []
        cam_distances_ = []
        for bi in range(B):
            poses_.append(poses[bi][target_indices[bi]])
            Ks_.append(Ks[bi][target_indices[bi]])
            cam_distances_.append(cam_distances[bi][target_indices[bi]])
        cam_distances_ = torch.stack(cam_distances_).reshape(B*TN, 1)
        poses_ = torch.stack(poses_).reshape(B*TN, 3, 4)
        Ks_ = torch.stack(Ks_).reshape(B*TN, 4, 4)
        
        near = torch.ones(B * TN, 1, H, W, dtype=spatial_volume.dtype, device=spatial_volume.device) * cam_distances_.unsqueeze(-1).unsqueeze(-1) - self.frustum_volume_length
        far = torch.ones(B * TN, 1, H, W, dtype=spatial_volume.dtype, device=spatial_volume.device) * cam_distances_.unsqueeze(-1).unsqueeze(-1) + self.frustum_volume_length
        # pdb.set_trace()
        # target_indices = target_indices.view(B*TN) # B*TN
        volume_xyz, volume_depth = create_target_volume(D, self.frustum_volume_size, self.input_image_size, poses_, Ks_, near, far, self.projection) # B*TN,3,D,H,W; B*TN,1,D,H,W
        # pdb.set_trace()
        
        # xyz = volume_xyz[0].permute(1,2,3,0).reshape(-1,3).detach().cpu().numpy()
        # np.save('xyz.npy', xyz)
        # pdb.set_trace()
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(volume_xyz.permute(0,2,3,4,1)[0].reshape(-1,3).detach().cpu().numpy())
        # o3d.io.write_point_cloud("./volume2.ply", pcd)
        # exit()
        volume_xyz_ = volume_xyz / self.spatial_volume_length  # since the spatial volume is constructed in [-spatial_volume_length,spatial_volume_length]
        volume_xyz_ = volume_xyz_.permute(0, 2, 3, 4, 1)  # B*TN,D,H,W,3
        spatial_volume_ = spatial_volume.unsqueeze(1).repeat(1, TN, 1, 1, 1, 1).view(B * TN, -1, V, V, V)
        volume_feats = F.grid_sample(spatial_volume_, volume_xyz_, mode='bilinear', padding_mode='zeros', align_corners=True) # B*TN,C,D,H,W

        v_embed_ = v_embed[torch.arange(B)[:,None], target_indices.view(B,TN)].view(B*TN, -1) # B*TN, v_dim
        t_embed_ = t_embed.unsqueeze(1).repeat(1,TN,1).view(B*TN,-1) # B*TN, t_dim
        volume_feats_dict = self.frustum_volume_feats(volume_feats, t_embed_, v_embed_) # 32: B*TN, F, D, H, W
        return volume_feats_dict, volume_depth

class SyncMultiviewDiffusion(pl.LightningModule):
    def __init__(self, unet_config, scheduler_config,
                 finetune_unet=False, finetune_projection=True, projection='perspective', use_spatial_volume=False,
                 view_num=16, image_size=256,
                 cfg_scale=3.0, output_num=8, batch_view_num=4,
                 drop_conditions=False, drop_scheme='default',
                 clip_image_encoder_path="/apdcephfs/private_rondyliu/projects/clip/ViT-L-14.pt",
                 sample_type='ddim', sample_steps=50, target_elevation=30):
        super().__init__()

        self.finetune_unet = finetune_unet
        self.finetune_projection = finetune_projection

        self.view_num = view_num
        self.viewpoint_dim = 4
        self.output_num = output_num
        self.image_size = image_size

        self.batch_view_num = batch_view_num
        self.cfg_scale = cfg_scale

        self.clip_image_encoder_path = clip_image_encoder_path
        self.target_elevation = target_elevation

        self._init_time_step_embedding()
        self._init_first_stage()
        self._init_schedule()
        self._init_clip_image_encoder()
        # self._init_clip_projection()
        self.spatial_volume = SpatialVolumeNet(self.time_embed_dim, self.viewpoint_dim, self.view_num, projection=projection, use_spatial_volume=use_spatial_volume)
        self.model = UNetWrapper(unet_config, drop_conditions=drop_conditions, drop_scheme=drop_scheme)
        self.scheduler_config = scheduler_config

        latent_size = image_size//8
        if sample_type=='ddim':
            self.sampler = SyncDDIMSampler(self, sample_steps , "uniform", 1.0, latent_size=latent_size)
        else:
            raise NotImplementedError

    def _init_clip_projection(self):
        self.cc_projection = nn.Linear(772, 768)
        nn.init.eye_(list(self.cc_projection.parameters())[0][:768, :768])
        nn.init.zeros_(list(self.cc_projection.parameters())[1])
        self.cc_projection.requires_grad_(True)

        if not self.finetune_projection:
            disable_training_module(self.cc_projection)

    def _init_multiview(self):
        K, azs, _, _, poses = read_pickle(f'assets/thuman_meta.pkl')
        default_image_size = 256
        ratio = self.image_size/default_image_size
        K = np.diag([ratio,ratio,1]) @ K[:3,:3]
        K = torch.from_numpy(K.astype(np.float32)) # [3,3]
        K = K.unsqueeze(0).repeat(self.view_num,1,1)        # N,3,3
        poses = torch.from_numpy(poses.astype(np.float32))  # N,3,4
        self.register_buffer('poses', poses)
        self.register_buffer('Ks', K)
        azs = (azs + np.pi) % (np.pi * 2) - np.pi # scale to [-pi,pi] and the index=0 has az=0
        self.register_buffer('azimuth', torch.from_numpy(azs.astype(np.float32)))
    
    def get_viewpoint_embedding(self, batch):
        """
        @param batch_size:
        @param elevation_ref: B
        @return:
        """
        azimuth_input = torch.deg2rad(batch['input_azimuth']) # 1
        azimuth_target = torch.deg2rad(batch['target_azimuth']) # N
        elevation_input = torch.deg2rad(batch['input_elevation']) # note that zero123 use a negative elevation here!!!
        elevation_target = torch.deg2rad(batch['target_elevation'])
        d_e = elevation_target - elevation_input
        d_a = azimuth_target - azimuth_input
        d_z = torch.zeros_like(d_a)
        embedding = torch.stack([d_e, torch.sin(d_a), torch.cos(d_a), d_z], -1) # B,N,4
        return embedding

    def _init_first_stage(self):
        first_stage_config={
            "target": "ldm.models.autoencoder.AutoencoderKL",
            "params": {
                "embed_dim": 4,
                "monitor": "val/rec_loss",
                "ddconfig":{
                  "double_z": True,
                  "z_channels": 4,
                  "resolution": self.image_size,
                  "in_channels": 3,
                  "out_ch": 3,
                  "ch": 128,
                  "ch_mult": [1,2,4,4],
                  "num_res_blocks": 2,
                  "attn_resolutions": [],
                  "dropout": 0.0
                },
                "lossconfig": {"target": "torch.nn.Identity"},
            }
        }
        self.first_stage_scale_factor = 0.18215
        self.first_stage_model = instantiate_from_config(first_stage_config)
        self.first_stage_model = disable_training_module(self.first_stage_model)

    def _init_clip_image_encoder(self):
        self.clip_image_encoder = FrozenCLIPImageEmbedder(model=self.clip_image_encoder_path)
        self.clip_image_encoder = disable_training_module(self.clip_image_encoder)

    def _init_schedule(self):
        self.num_timesteps = 1000
        linear_start = 0.00085
        linear_end = 0.0120
        num_timesteps = 1000
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2 # T
        assert betas.shape[0] == self.num_timesteps

        # all in float64 first
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # T
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # T
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_log_variance_clipped = torch.clamp(posterior_log_variance_clipped, min=-10)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).float())
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped.float())

    def _init_time_step_embedding(self):
        self.time_embed_dim = 256
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(True),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

    def encode_first_stage(self, x, sample=True):
        with torch.no_grad():
            posterior = self.first_stage_model.encode(x)  # b,4,h//8,w//8
            if sample:
                return posterior.sample().detach() * self.first_stage_scale_factor
            else:
                return posterior.mode().detach() * self.first_stage_scale_factor

    def decode_first_stage(self, z):
        with torch.no_grad():
            z = 1. / self.first_stage_scale_factor * z
            return self.first_stage_model.decode(z)

    def prepare(self, batch):
        # encode target
        if 'target_image' in batch:
            image_target = batch['target_image'].permute(0, 1, 4, 2, 3) # b,n,3,h,w
            N = image_target.shape[1]
            x = [self.encode_first_stage(image_target[:,ni], True) for ni in range(N)]
            x = torch.stack(x, 1) # b,n,4,h//8,w//8
        else:
            x = None

        image_input = batch['input_image'].permute(0, 3, 1, 2)
        elevation_input = batch['input_elevation'][:, 0] # b
        x_input = self.encode_first_stage(image_input)
        input_info = {'image': image_input, 'elevation': elevation_input, 'x': x_input}
        with torch.no_grad():
            clip_embed = self.clip_image_encoder.encode(image_input)
        return x, clip_embed, input_info

    def embed_time(self, t):
        t_embed = timestep_embedding(t, self.time_embed_dim, repeat_only=False) # B,TED
        t_embed = self.time_embed(t_embed) # B,TED
        return t_embed

    def get_target_view_feats(self, x_input, spatial_volume, clip_embed, t_embed, v_embed, target_index, batch):
        """
        @param x_input:        B,4,H,W
        @param spatial_volume: B,C,V,V,V
        @param clip_embed:     B,1,768
        @param t_embed:        B,t_dim
        @param v_embed:        B,N,v_dim
        @param target_index:   B,TN
        @return:
            tensors of size B*TN,*
        """
        B, _, H, W = x_input.shape
        frustum_volume_feats, frustum_volume_depth = self.spatial_volume.construct_view_frustum_volume(spatial_volume, t_embed, v_embed, target_index, batch)
        # clip
        TN = target_index.shape[1] # number of views in a batch
        v_embed_ = v_embed[torch.arange(B)[:,None], target_index].view(B*TN, self.viewpoint_dim) # B*TN,v_dim
        clip_embed_ = clip_embed.unsqueeze(1).repeat(1,TN,1,1).view(B*TN,1,768)
        # clip_embed_ = self.cc_projection(torch.cat([clip_embed_, v_embed_.unsqueeze(1)], -1))  # B*TN,1,768

        x_input_ = x_input.unsqueeze(1).repeat(1, TN, 1, 1, 1).view(B * TN, 4, H, W)

        x_concat = x_input_
        return clip_embed_, frustum_volume_feats, x_concat

    def training_step(self, batch):
        B = batch['target_image'].shape[0]
        time_steps = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()

        x, clip_embed, input_info = self.prepare(batch)
        x_noisy, noise = self.add_noise(x, time_steps)  # B,N,4,H,W

        N = self.view_num
        target_index = torch.randint(0, N, (B, 1), device=self.device).long() # B, 1
        v_embed = self.get_viewpoint_embedding(batch) # N,v_dim
        t_embed = self.embed_time(time_steps)
        spatial_volume = self.spatial_volume.construct_spatial_volume(x_noisy, t_embed, v_embed, batch)

        clip_embed, volume_feats, x_concat = self.get_target_view_feats(input_info['x'], spatial_volume, clip_embed, t_embed, v_embed, target_index, batch)

        x_noisy_ = x_noisy[torch.arange(B)[:,None],target_index][:,0] # B,4,H,W
        # pdb.set_trace()
        noise_predict = self.model(x_noisy_, time_steps, clip_embed, volume_feats, x_concat, is_train=True) # B,4,H,W

        noise_target = noise[torch.arange(B)[:,None],target_index][:,0] # B,4,H,W
        # loss simple for diffusion
        loss_simple = torch.nn.functional.mse_loss(noise_target, noise_predict, reduction='none')
        loss = loss_simple.mean()
        self.log('sim', loss_simple.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True, rank_zero_only=True)

        # log others
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        self.log("step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        return loss

    def add_noise(self, x_start, t):
        """
        @param x_start: B,*
        @param t:       B,
        @return:
        """
        B = x_start.shape[0]
        noise = torch.randn_like(x_start) # B,*

        sqrt_alphas_cumprod_  = self.sqrt_alphas_cumprod[t] # B,
        sqrt_one_minus_alphas_cumprod_ = self.sqrt_one_minus_alphas_cumprod[t] # B
        sqrt_alphas_cumprod_ = sqrt_alphas_cumprod_.view(B, *[1 for _ in range(len(x_start.shape)-1)])
        sqrt_one_minus_alphas_cumprod_ = sqrt_one_minus_alphas_cumprod_.view(B, *[1 for _ in range(len(x_start.shape)-1)])
        x_noisy = sqrt_alphas_cumprod_ * x_start + sqrt_one_minus_alphas_cumprod_ * noise
        return x_noisy, noise

    def sample(self, sampler, batch, cfg_scale, batch_view_num, return_inter_results=False, inter_interval=50, inter_view_interval=2):
        _, clip_embed, input_info = self.prepare(batch)
        x_sample, inter = sampler.sample(input_info, clip_embed, unconditional_scale=cfg_scale, log_every_t=inter_interval, batch_view_num=batch_view_num, batch=batch)

        N = x_sample.shape[1]
        x_sample = torch.stack([self.decode_first_stage(x_sample[:, ni]) for ni in range(N)], 1)
        if return_inter_results:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            inter = torch.stack(inter['x_inter'], 2) # # B,N,T,C,H,W
            B,N,T,C,H,W = inter.shape
            inter_results = []
            for ni in tqdm(range(0, N, inter_view_interval)):
                inter_results_ = []
                for ti in range(T):
                    inter_results_.append(self.decode_first_stage(inter[:, ni, ti]))
                inter_results.append(torch.stack(inter_results_, 1)) # B,T,3,H,W
            inter_results = torch.stack(inter_results,1) # B,N,T,3,H,W
            return x_sample, inter_results
        else:
            return x_sample

    def log_image(self,  x_sample, batch, step, output_dir):
        process = lambda x: ((torch.clip(x, min=-1, max=1).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        B = x_sample.shape[0]
        N = x_sample.shape[1]
        image_cond = []
        for bi in range(B):
            img_pr_ = concat_images_list(process(batch['input_image'][bi]),*[process(x_sample[bi, ni].permute(1, 2, 0)) for ni in range(N)])
            image_cond.append(img_pr_)

        output_dir = Path(output_dir)
        imsave(str(output_dir/f'{step}.jpg'), concat_images_list(*image_cond, vert=True))

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if batch_idx==0 and self.global_rank==0:
            self.eval()
            step = self.global_step
            batch_ = {}
            for k, v in batch.items():
                if isinstance(batch[k], dict):
                    batch_[k] = {}
                    for k_, v_ in batch[k].items():
                        batch_[k][k_] = v_[:self.output_num]
                else:
                    batch_[k] = v[:self.output_num]
            x_sample = self.sample(self.sampler, batch_, self.cfg_scale, self.batch_view_num)
            output_dir = Path(self.image_dir) / 'images' / 'val'
            output_dir.mkdir(exist_ok=True, parents=True)
            self.log_image(x_sample, batch, step, output_dir=output_dir)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.eval()
        x_sample = self.sample(self.sampler, batch, self.cfg_scale, self.batch_view_num)
        output_dir = Path(self.outdir)
        output_dir.mkdir(exist_ok=True, parents=True)
        self.log_image(x_sample, batch, batch_idx, output_dir=output_dir)
    
    def configure_optimizers(self):
        lr = self.learning_rate
        print(f'setting learning rate to {lr:.4f} ...')
        paras = []
        # if self.finetune_projection:
        #     paras.append({"params": self.cc_projection.parameters(), "lr": lr},)
        if self.finetune_unet:
            paras.append({"params": self.model.parameters(), "lr": lr},)
        else:
            paras.append({"params": self.model.get_trainable_parameters(), "lr": lr},)

        paras.append({"params": self.time_embed.parameters(), "lr": lr*10.0},)
        paras.append({"params": self.spatial_volume.parameters(), "lr": lr*10.0},)

        opt = torch.optim.AdamW(paras, lr=lr)

        scheduler = instantiate_from_config(self.scheduler_config)
        print("Setting up LambdaLR scheduler...")
        scheduler = [{'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule), 'interval': 'step', 'frequency': 1}]
        return [opt], scheduler

class SyncDDIMSampler:
    def __init__(self, model: SyncMultiviewDiffusion, ddim_num_steps, ddim_discretize="uniform", ddim_eta=1.0, latent_size=32, optimize_latent=False):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.latent_size = latent_size
        self._make_schedule(ddim_num_steps, ddim_discretize, ddim_eta)
        self.eta = ddim_eta
        self.optimize_latent = optimize_latent

    def _make_schedule(self,  ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps, num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose) # DT
        ddim_timesteps_ = torch.from_numpy(self.ddim_timesteps.astype(np.int64)) # DT

        alphas_cumprod = self.model.alphas_cumprod # T
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        self.ddim_alphas = alphas_cumprod[ddim_timesteps_].double() # DT
        self.ddim_alphas_prev = torch.cat([alphas_cumprod[0:1], alphas_cumprod[ddim_timesteps_[:-1]]], 0) # DT
        self.ddim_sigmas = ddim_eta * torch.sqrt((1 - self.ddim_alphas_prev) / (1 - self.ddim_alphas) * (1 - self.ddim_alphas / self.ddim_alphas_prev))

        self.ddim_alphas_raw = self.model.alphas[ddim_timesteps_].float() # DT
        self.ddim_sigmas = self.ddim_sigmas.float()
        self.ddim_alphas = self.ddim_alphas.float()
        self.ddim_alphas_prev = self.ddim_alphas_prev.float()
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1. - self.ddim_alphas).float()

    # @torch.no_grad()
    def denoise_apply_impl(self, x_target_noisy, index, noise_pred, is_step0=False):
        """
        @param x_target_noisy: B,N,4,H,W
        @param index:          index
        @param noise_pred:     B,N,4,H,W
        @param is_step0:       bool
        @return:
        """
        device = x_target_noisy.device
        B,N,_,H,W = x_target_noisy.shape

        # apply noise
        a_t = self.ddim_alphas[index].to(device).float().view(1,1,1,1,1)
        a_prev = self.ddim_alphas_prev[index].to(device).float().view(1,1,1,1,1)
        sqrt_one_minus_at = self.ddim_sqrt_one_minus_alphas[index].to(device).float().view(1,1,1,1,1)
        sigma_t = self.ddim_sigmas[index].to(device).float().view(1,1,1,1,1)

        pred_x0 = (x_target_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt()
        dir_xt = torch.clamp(1. - a_prev - sigma_t**2, min=1e-7).sqrt() * noise_pred
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt
        if not is_step0:
            noise = sigma_t * torch.randn_like(x_target_noisy)
            x_prev = x_prev + noise
        return x_prev

    # @torch.no_grad()
    def denoise_apply(self, x_target_noisy, input_info, clip_embed, time_steps, index, unconditional_scale, batch_view_num=1, is_step0=False, batch=None):
        """
        @param x_target_noisy:   B,N,4,H,W
        @param input_info:
        @param clip_embed:       B,M,768
        @param time_steps:       B,
        @param index:            int
        @param unconditional_scale:
        @param batch_view_num:   int
        @param is_step0:         bool
        @return:
        """
        x_input, elevation_input = input_info['x'], input_info['elevation']
        B, N, C, H, W = x_target_noisy.shape

        # construct source data
        v_embed = self.model.get_viewpoint_embedding(batch) # B,N,v_dim
        t_embed = self.model.embed_time(time_steps)  # B,t_dim
        spatial_volume = self.model.spatial_volume.construct_spatial_volume(x_target_noisy, t_embed, v_embed, batch) # B,64,32,32,32; fused spatial features for all 16 views

        e_t = []
        target_indices = torch.arange(N) # N
        for ni in range(0, N, batch_view_num):
            x_target_noisy_ = x_target_noisy[:, ni:ni + batch_view_num]
            VN = x_target_noisy_.shape[1]
            x_target_noisy_ = x_target_noisy_.reshape(B*VN,C,H,W)

            time_steps_ = repeat_to_batch(time_steps, B, VN)
            target_indices_ = target_indices[ni:ni+batch_view_num].unsqueeze(0).repeat(B,1)
            clip_embed_, volume_feats_, x_concat_ = self.model.get_target_view_feats(x_input, spatial_volume, clip_embed, t_embed, v_embed, target_indices_, batch)
            if unconditional_scale!=1.0:
                noise = self.model.model.predict_with_unconditional_scale(x_target_noisy_, time_steps_, clip_embed_, volume_feats_, x_concat_, unconditional_scale)
            else:
                noise = self.model.model(x_target_noisy_, time_steps_, clip_embed_, volume_feats_, x_concat_, is_train=False)
            e_t.append(noise.view(B,VN,4,H,W))

        e_t = torch.cat(e_t, 1)
        x_prev = self.denoise_apply_impl(x_target_noisy, index, e_t, is_step0)
        return x_prev
    
    # @torch.no_grad()
    def sample(self, input_info, clip_embed, unconditional_scale=1.0, log_every_t=50, batch_view_num=1, batch=None):
        """
        @param input_info:      x, elevation
        @param clip_embed:      B,M,768
        @param unconditional_scale:
        @param log_every_t:
        @param batch_view_num:
        @return:
        """
        print(f"unconditional scale {unconditional_scale:.1f}")
        C, H, W = 4, self.latent_size, self.latent_size
        B = clip_embed.shape[0]
        N = self.model.view_num
        device = self.model.device
        x_target_noisy = torch.randn([B, N, C, H, W], requires_grad=True, device=device)

        timesteps = self.ddim_timesteps
        intermediates = {'x_inter': []}
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        
        with torch.no_grad():
            timesteps_50 = self.ddim_timesteps
            time_range = np.flip(timesteps_50)
            total_steps = timesteps_50.shape[0]
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
            for i, step in enumerate(iterator):
                index = total_steps - i - 1 # index in ddim state
                time_steps = torch.full((B,), step, device=device, dtype=torch.long)
                x_target_noisy = self.denoise_apply(x_target_noisy, input_info, clip_embed, time_steps, index, unconditional_scale, batch_view_num=batch_view_num, is_step0=index==0, batch=batch)
                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates['x_inter'].append(x_target_noisy)
        
        
        return x_target_noisy, intermediates