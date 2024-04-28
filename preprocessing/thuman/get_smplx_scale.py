import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot
import argparse
import os

def main(args):
    os.makedirs(f'{args.output_dir}/smplx_stats/', exist_ok=True)
    for i in range(2445):
        uid = str(i).zfill(4)
        smpl_params = np.load(f'{args.smplx_dir}/{uid}/smplx_param.pkl', allow_pickle=True)
        
        scale = 0.6/smpl_params['scale'][0]
        v = trimesh.load(f'{args.smplx_dir}/{uid}/mesh_smplx.obj', process=False).vertices
        Rot_90 = Rot.from_euler('xyz', [90,0,0],True).as_matrix()
        if int(i) < 526:
            v = (Rot_90@v.T).T
        center = (v.min(axis=0)+v.max(axis=0))/2 * scale
        np.save(f'{args.output_dir}/smplx_stats/{uid}.npy', np.array([scale] + list(center)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_dir", type=str, default='./assets/thuman_smplx')
    parser.add_argument("--output_dir", type=str, default='/cluster/scratch/xiychen/data/thuman_2.1_preprocessed')
    args = parser.parse_args()
    main(args)