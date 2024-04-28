"""
script to process the FaceScape Dataset (apply square cropping, resizing and color calibration)
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

import os
import json
import numpy as np
import trimesh
import cv2
import renderer
import tqdm
import argparse
import openmesh
from calibrate_colors import calibrate_colors
import glob

def to_homogeneous_trafo(trafo: np.ndarray):
    """

    :param trafo: N, 3, 4
    :return: trafo N, 4, 4 (appended [0,0,0,1])
    """
    return np.concatenate((trafo, np.tile(np.array([[[0, 0, 0, 1.]]]), (len(trafo), 1, 1))), axis=1)


def read_cam_extrinsics(cam_dict):
    extrinsics = []
    i = 0
    while True:
        try:
            extrinsics.append(cam_dict[f"{i}_Rt"])
            i += 1
        except KeyError:
            break
    extrinsics = np.array(extrinsics)
    extrinsics = to_homogeneous_trafo(extrinsics)
    return extrinsics


def get_cam_angles(Rt, ref_dir=np.array([0., 1., 0.])):
    cam_viewdir = Rt[2, :3]

    cam_viewdir_hor = np.copy(cam_viewdir)
    cam_viewdir_hor[2] = 0
    cam_viewdir_hor /= np.sum(cam_viewdir_hor ** 2) ** .5
    cam_viewdir_vert = np.copy(cam_viewdir)
    cam_viewdir_vert[0] = 0
    cam_viewdir_vert /= np.sum(cam_viewdir_vert ** 2) ** .5

    azimuth = np.arccos(np.matmul(cam_viewdir_hor, ref_dir)) * 180. / np.pi
    elevation = np.arccos(np.matmul(cam_viewdir_vert, ref_dir)) * 180. / np.pi

    azimuth *= -1 * np.sign(cam_viewdir_hor[0])
    elevation *= np.sign(cam_viewdir_vert[2])

    return dict(azimuth=azimuth, elevation=elevation)


def inv_extrinsics(extr):
    """

    :param extr:  N x 4 x 4
    :return:
    """

    R = extr[:, :3, :3]
    T = extr[:, :3, -1:]

    R_inv = R.transpose(0, 2, 1)
    T_inv = -R_inv @ T
    extr_inv = np.concatenate((R_inv, T_inv), axis=-1)
    extr_inv = to_homogeneous_trafo(extr_inv)
    return extr_inv


UINT16_MAX = 65535
SCALE_FACTOR = 1e-4  # corresponds to a representative power of 6.535m with .1 mm resolution


def float32_2_uint16(x):
    float_max = UINT16_MAX * SCALE_FACTOR
    return (x.clip(max=float_max) / SCALE_FACTOR).round().astype(np.uint16)


with open("../../assets/Rt_scale_dict.json", "r") as f:
    align_Rts_dict = json.load(f)

# lm_list_v10 = np.load("assets/facescape/landmark_indices.npz")["v10"]
FACESCAPE_2_CAPSTUDIO = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0]])
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def main(in_subject_root,
         out_subject_root,
         crop_out=256,
         padding_v=.01,
         padding_h=.05,
         save_bilinear_vertices=False,
         save_depth=False):
    pose_dirs = sorted([d for d in in_subject_root.iterdir() if d.is_dir() and d.name[0].isnumeric()])
    for pose_dir in tqdm.tqdm(pose_dirs, desc="Poses"):
        s_idx = in_subject_root.name
        p_idx = pose_dir.name.split("_")[0]

        # try:
        with open(pose_dir / "params.json", "r") as f:
            cam_dict = json.load(f)
        extrinsics = read_cam_extrinsics(cam_dict)
        mesh = trimesh.load(pose_dir.parent / (pose_dir.name + ".ply"))
        om_mesh = openmesh.read_trimesh(str(pose_dir.parent / "models_reg" / (pose_dir.name + ".obj")))
        verts_3d = om_mesh.points()
        del om_mesh
        # except Exception as e:
        #     print("ERROR", e)
        #     continue

        poses = inv_extrinsics(extrinsics)
        scale_align = align_Rts_dict[s_idx][p_idx][0]
        Rt_align = np.array(align_Rts_dict[s_idx][p_idx][1])
        Rt_align = to_homogeneous_trafo(Rt_align[None])[0]

        # adopt camera convention:
        # up: z, left side of head: x, face looks in negative y direction
        # 1 unit: 1m
        Rt_align[:3] = FACESCAPE_2_CAPSTUDIO @ Rt_align[:3]
        poses[:, :3, -1] *= scale_align
        poses = np.tile(Rt_align[None], (len(extrinsics), 1, 1)) @ poses
        poses[:, :3, -1] /= 1000
        extrinsics = inv_extrinsics(poses)
        mesh.vertices *= scale_align
        mesh.vertices = np.tensordot(Rt_align[:3, :3], mesh.vertices.T, 1).T + Rt_align[:3, 3]
        mesh.vertices /= 1000
        # lmk_3d = (FACESCAPE_2_CAPSTUDIO @ lmk_3d.T).T
        # lmk_3d /= 1000

        # # visualize scene geometry
        # import matplotlib.pyplot as plt
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # s = .3
        # for i, c in enumerate(["red", "green", "blue"]):
        #     ax.quiver(poses[:, 0, -1], poses[:, 1, -1], poses[:, 2, -1],
        #               s * poses[:, 0, i], s * poses[:, 1, i], s * poses[:, 2, i], edgecolor=c)
        # for i in range(len(poses)):
        #     ax.text(poses[i, 0, -1], poses[i, 1, -1], poses[i, 2, -1], str(i))
        # idcs = np.random.choice(np.arange(len(mesh.vertices)), 1000)
        # ax.scatter(mesh.vertices[idcs, 0], mesh.vertices[idcs, 1], mesh.vertices[idcs, 2])
        # ax.scatter(lmk_3d[:, 0], lmk_3d[:, 1], lmk_3d[:, 2])
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        # plt.show()

        cam_outdict = dict()

        # image undistorting, cropping, intrinsics correction
        view_dirs = sorted([f for f in pose_dir.iterdir() if not f.name.endswith(".json")])
        for img in tqdm.tqdm(view_dirs, leave=False, desc="Views"):
            # try:
            i_idx = img.name.split(".")[0]
            K = np.array(cam_dict[i_idx + "_K"])
            Rt = extrinsics[int(i_idx), :3]
            angles = get_cam_angles(Rt)
            if abs(angles['azimuth']) > 90:
                continue
            pose = poses[int(i_idx)]
            distortion = np.array(cam_dict[i_idx + "_distortion"])
            w = cam_dict[i_idx + "_width"]
            h = cam_dict[i_idx + "_height"]
            valid = cam_dict[i_idx + "_valid"]

            if valid:
                rgb = cv2.imread(str(img))
                rgb = cv2.undistort(rgb, K, distortion)
                depth, color = renderer.render_cvcam(mesh, K, Rt, rend_size=(h, w))
                mask = depth > 0

                # cropping image:
                crop_in = min(h, w)
                padding_px_v = int(crop_in * padding_v)
                padding_px_h = int(crop_in * padding_h)

                # getting bbx
                fg_y, fg_x = np.where(mask)
                silh_top = np.min(fg_y)
                silh_bottom = np.max(fg_y)
                silh_left = np.min(fg_x)
                silh_right = np.max(fg_x)

                cam_center = pose[:3, -1]
                if cam_center[0] < 0:  # cam on right head side -> orientation wrt. right silhouette end
                    bbx_top = np.clip(silh_top - padding_px_v, a_min=0, a_max=None)
                    bbx_right = np.clip(silh_right + padding_px_h, a_max=w, a_min=None)
                    bbx_bottom = np.clip(bbx_top + crop_in, a_max=h, a_min=None)
                    bbx_left = np.clip(bbx_right - crop_in, a_min=0, a_max=None)
                    bbx_top = bbx_bottom - crop_in
                    bbx_right = bbx_left + crop_in

                else:  # cam on left head side -> orientation wrt. left silhouette end
                    bbx_top = np.clip(silh_top - padding_px_v, a_min=0, a_max=None)
                    bbx_left = np.clip(silh_left - padding_px_h, a_min=0, a_max=None)
                    bbx_bottom = np.clip(bbx_top + crop_in, a_max=h, a_min=None)
                    bbx_right = np.clip(bbx_left + crop_in, a_max=w, a_min=None)
                    bbx_top = bbx_bottom - crop_in
                    bbx_left = bbx_right - crop_in

                # applying bbx
                rgb = rgb[bbx_top:bbx_bottom, bbx_left:bbx_right]

                depth = depth[bbx_top:bbx_bottom, bbx_left:bbx_right]
                mask = mask[bbx_top:bbx_bottom, bbx_left:bbx_right]
                K[0, -1] -= bbx_left
                K[1, -1] -= bbx_top

                # scaling
                rgb = cv2.resize(rgb, (crop_out, crop_out), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, (crop_out, crop_out), interpolation=cv2.INTER_NEAREST)
                mask = depth > 0
                K[:2] *= crop_out / crop_in

                # writing view-specific output
                rgba = np.concatenate((rgb, mask.astype(int)[..., None] * 255), axis=-1)
                outdir = out_subject_root / f"{int(p_idx):02d}" / f"view_{int(i_idx):05d}"
                os.makedirs(outdir, exist_ok=True)
                # print(str(outdir / "rgba.png"))
                cv2.imwrite(str(outdir / "rgba.png"), rgba)
                if save_depth:
                    depth = float32_2_uint16(depth)
                    cv2.imwrite(str(outdir / "depth.png"), depth)

                cam_outdict[int(i_idx)] = dict(
                    intrinsics=K.tolist(),
                    extrinsics=Rt.tolist(),
                    angles=angles
                )

        out_scan_dir = out_subject_root / f"{int(p_idx):02d}"
        if save_bilinear_vertices:
            np.savetxt(out_scan_dir / "face_vertices.npy", verts_3d)
        with open(out_scan_dir / "cameras.json", "w") as f:
            json.dump(cam_outdict, f)

        # color calibration
        calibrate_colors(out_scan_dir, mesh=mesh)
        rgbas = glob.glob(os.path.join(out_scan_dir, 'view_*', 'rgba.png'))
        # remove uncalibrated images to save disk space
        for f in rgbas:
            os.remove(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_in", type=Path,
                        help="Input subject directory e.g. 'FACESCAPE_RAW/1'")
    parser.add_argument("--dir_out", type=Path,
                        help="Output subject directory e.g. 'FACESCAPE_PROCESSED/001'")
    parser.add_argument("--tmp_dir", type=Path, default=Path("/scratch"),
                        help="temporary directory to perform the conversion on. Useful when processing the dataset in parallel "
                             "on a compute cluster to avoid io overloads.")
    parser.add_argument("--crop_out", type=int, default=256)
    parser.add_argument("--padding_v", type=float, default=.01)
    parser.add_argument("--padding_h", type=float, default=.05)
    parser.add_argument("--save_bilinear_vertices", type=bool, default=False)
    args = parser.parse_args()
    tmp_in = args.tmp_dir / "in" / args.dir_in.name
    tmp_out = args.tmp_dir / "out" / args.dir_out.name
    tmp_in.parent.mkdir(parents=True, exist_ok=True)
    tmp_out.parent.mkdir(parents=True, exist_ok=True)
    args.dir_out.parent.mkdir(parents=True, exist_ok=True)
    os.system(f"cp -r {args.dir_in} {tmp_in.parent}")
    main(tmp_in, args.dir_out, crop_out=args.crop_out, padding_h=args.padding_h, padding_v=args.padding_v, save_bilinear_vertices=args.save_bilinear_vertices)
    # os.system(f"rm -r {args.dir_out}")
    # os.system(f"cp -r {tmp_out} {args.dir_out}")