from pathlib import Path
import numpy as np
import json
import sys
import renderer
import matplotlib.pyplot as plt
import trimesh
import torch
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import tqdm
from sklearn import linear_model
from tenacity import retry, stop_after_delay, wait_fixed
from torchvision.utils import save_image
import os
from argparse import ArgumentParser


def retry_f(f, *args, retry_kwargs=dict(stop=stop_after_delay(30), wait=wait_fixed(10)), **kwargs):
    def before_sleep(state):
        print("Error Occured:", state.outcome._exception)
        print(f"Retrying...")

    @retry(**retry_kwargs, before_sleep=before_sleep)
    def retry_wrapper():
        return f(*args, **kwargs)

    return retry_wrapper()


def calibrate_colors(root,
                     rgb_in_fname="rgba.png",
                     rgb_out_fname="rgba_colorcalib.png",
                     mesh = "mesh.obj",
                     ncams=-1,
                     specular_thr=.7,
                     l1_thr=0.085,
                     red_outlier_thr=.3,
                     red_outlier_ratio_thr=.03,
                     optimizer_kwargs=dict(epsilon=1., alpha=0., max_iter=1000, tol=1e-6),
                     device=torch.device("cpu"),
                     verbose=False):
    cam_path = root / "cameras.json"
    if isinstance(mesh, str):
        mesh_path = root / mesh
        mesh = retry_f(trimesh.load, str(mesh_path))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with retry_f(open, cam_path, "r") as f:
        cam_dict = json.load(f)
    ncams = ncams if ncams > 0 else len(cam_dict.keys())


    nverts = len(mesh.vertices)

    all_vert_colors = []
    all_vert_idcs = []
    cam_ids = np.array(sorted(cam_dict.keys()))
    if ncams != len(cam_dict.keys()):
        cam_ids = np.random.choice(cam_ids, ncams, replace=False)
    unsuccessful_cam_idcs = []
    for i, camid in enumerate(tqdm.tqdm(cam_ids, desc="Color Calibration", leave=False)):
        # try:
        img_path = root / f"view_{int(camid):05d}" / rgb_in_fname
        rgb = retry_f(pil_to_tensor, retry_f(Image.open, img_path))[:3][None].float().to(device) / 255.
        h, w = rgb.shape[-2:]
        K = cam_dict[camid]["intrinsics"]
        Rt = cam_dict[camid]["extrinsics"]
        depth, _ = renderer.render_cvcam(mesh, K, Rt, rend_size=(h, w))

        # transforming vertices into camera frame
        K = torch.tensor(K).float().to(device)
        world_2_cam = torch.cat((torch.tensor(Rt), torch.tensor([[0., 0., 0., 1]])), dim=0).to(device)
        verts_world = torch.from_numpy(mesh.vertices).float().to(device)
        verts_world = torch.cat((verts_world, torch.ones_like(verts_world[:, :1])), dim=-1)
        verts_cam = (K @ (world_2_cam @ verts_world.T)[:3]).T
        verts_cam[:, :2] = verts_cam[:, :2] / verts_cam[:, 2:]  # (N, 3), (u, v, z)
        verts_cam[:, :2] = verts_cam[:, :2] / torch.tensor([[w, h]], device=device) * 2 - 1
        # normalizing from 0 ... H/W to -1 ... +1

        # interpolating depth values
        depth = torch.from_numpy(depth).float().to(device)[None][None]  # (1, 1, H, W)
        sample_grid = verts_cam[:, :2][None][None]  # (1, 1, N, 2)
        verts_depth_sampled = torch.nn.functional.grid_sample(depth, sample_grid, mode="nearest",
                                                                padding_mode="zeros",
                                                                align_corners=False)
        verts_depth_sampled = verts_depth_sampled[0, 0, 0][:, None]  # (N, 1)
        verts_fg_mask = verts_depth_sampled[:, 0] != 0.

        # # visualize points after depth sampling
        # random_idcs = np.random.choice(np.where(verts_fg_mask.cpu().numpy())[0], 10000)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.scatter(verts_cam[random_idcs, 0].cpu(), verts_cam[random_idcs, 1].cpu(), verts_cam[random_idcs, 2].cpu(), s=5.)
        # ax.scatter(verts_cam[random_idcs, 0].cpu(), verts_cam[random_idcs, 1].cpu(), verts_depth_sampled[random_idcs, 0].cpu(), s=5.)
        # plt.show()

        # interpolating vertex colors
        sample_grid = verts_cam[:, :2].float()[None][None]  # (1, 1, N, 2)
        verts_color = torch.nn.functional.grid_sample(rgb, sample_grid, mode="bilinear", padding_mode="border",
                                                        align_corners=False)
        verts_color = verts_color[0, :, 0].T  # (N, 3)

        verts_visible_mask = (torch.abs(verts_depth_sampled - verts_cam[:, 2:]) < 0.003)[:, 0]
        specular_mask = verts_color.mean(dim=-1) >= specular_thr
        mask = verts_visible_mask & (~specular_mask)

        verts_masked_idcs = torch.where(mask)[0]
        nmasked = len(verts_masked_idcs)
        verts_masked_cam = verts_cam[mask]
        verts_masked_color = verts_color[mask]

        # # visualize point colors
        # random_idcs = np.random.choice(np.arange(nmasked), 10000)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.scatter(verts_masked_cam[random_idcs, 0].cpu(), verts_masked_cam[random_idcs, 1].cpu(), verts_masked_cam[random_idcs, 2].cpu(),
        #            s=5., c=verts_masked_color[random_idcs].cpu())
        # plt.show()

        # storing vertex colors
        all_vert_colors.append(verts_masked_color)
        all_vert_idcs.append(verts_masked_idcs)
        # except Exception as e:
        #     print(f"ERROR with {img_path}", e)
        #     unsuccessful_cam_idcs.append(i)

    # remove unsuccessful cam_ids
    if unsuccessful_cam_idcs:
        unsuccessful_cam_idcs = np.array(unsuccessful_cam_idcs, dtype=int)
        successful_mask = np.ones((len(cam_ids)), dtype=bool)
        successful_mask[unsuccessful_cam_idcs] = False
        cam_ids = cam_ids[successful_mask]
        ncams = len(cam_ids)

    # average_vert_colors
    mean_vert_colors = torch.zeros((nverts, 3), device=device, dtype=torch.float)
    vert_visibility_count = torch.zeros((nverts,), device=device, dtype=torch.int)
    for v_c, v_idx in zip(all_vert_colors, all_vert_idcs):
        mean_vert_colors[v_idx] += v_c
        vert_visibility_count[v_idx] += 1
    mean_vert_colors /= vert_visibility_count[:, None] + 1e-4

    # determine strong deviating cameras
    l1 = []
    red_outlier_ratios = []

    for v_c, v_idx in zip(all_vert_colors, all_vert_idcs):
        vertex_wise_l1 = torch.abs(mean_vert_colors[v_idx] - v_c)
        l1.append(vertex_wise_l1.mean().item())
        red_outlier_ratios.append(
            ((vertex_wise_l1[:, 0] > red_outlier_thr) & (torch.all(v_c < 50. / 255., dim=-1))).float().mean().item())

    l1 = np.array(l1)
    sorted_cam_idcs = np.argsort(l1)[::-1]
    sorted_cam_ids = cam_ids[sorted_cam_idcs]
    sorted_l1s = l1[sorted_cam_idcs]
    if verbose:
        print("Cam ids sorted by error (highest first)")
        [print(i, l) for i, l in zip(sorted_cam_ids, sorted_l1s)]

    # # Visualize images sorted by initial l1 error
    # n = int(np.ceil(np.sqrt(len(sorted_cam_idcs))))
    # fig, axes = plt.subplots(ncols=n, nrows=n, figsize=(3 * n, 3 * n))
    # axes = axes.flatten()
    # for i, idx in enumerate(sorted_cam_idcs):
    #     camid = cam_ids[idx]
    #     img_path = root / f"view_{int(camid):05d}" / rgb_in_fname
    #     axes[i].imshow(Image.open(img_path))
    #     axes[i].set_title(
    #         f"{'/'.join([p.name for p in list(img_path.parents)[:3]])}\nl1: {l1[idx]:.4e}\noutlier_ratio: {red_outlier_ratios[idx]}",
    #         fontsize=9)
    # [a.axis("off") for a in axes]
    # plt.show()

    # calculating affine color corrections
    # trying to solve for A to minimize |c_pred @ (A + Eye) - c_gt|^2_2 + alpha * |A|^2_2
    color_correctors = []  # (N, 3, 4)
    optimizer = linear_model.HuberRegressor(**optimizer_kwargs, fit_intercept=False, warm_start=False)
    for v_c, v_idx in tqdm.tqdm(zip(all_vert_colors, all_vert_idcs), desc="Solving Color Correction",
                                total=ncams, leave=False):
        y = mean_vert_colors[v_idx] - v_c  # (N, 3)
        y = y.cpu().numpy()
        X = v_c.cpu().numpy()
        X = np.concatenate((X, np.ones_like(X[:, :1])), axis=-1)
        A = []
        for i in range(3):
            optimizer.fit(X, y[:, i])
            A_ = optimizer.coef_
            A_[i] += 1.
            A.append(A_)
        A = np.stack(A, axis=0)
        color_correctors.append(torch.from_numpy(A).float().to(device))

    l1_corrected = []
    for v_c, v_idx, A in zip(all_vert_colors, all_vert_idcs, color_correctors):
        v_c_corrected = (A @ torch.cat((v_c, torch.ones_like(v_c[:, :1])), dim=-1).T).T
        l1_corrected.append(torch.abs(mean_vert_colors[v_idx] - v_c_corrected).mean().item())
    l1_corrected = np.array(l1_corrected)
    sorted_l1s_corrected = l1_corrected[sorted_cam_idcs]

    if verbose:
        print("Cam ids sorted by error (highest first), corrected error")
        [print(f"{i}, {l:.3f}, {l_corr:.3f}") for i, l, l_corr in
         zip(sorted_cam_ids, sorted_l1s, sorted_l1s_corrected)]

    # # Visualize colored mesh
    # colored_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_colors=mean_vert_colors.cpu().numpy())
    # from trimesh.viewer import SceneViewer
    # SceneViewer(trimesh.Scene(geometry=colored_mesh))

    # correcting images
    for idx in sorted_cam_idcs:
        # try:
        camid = cam_ids[idx]
        img_path = root / f"view_{int(camid):05d}" / rgb_in_fname
        out_path = root / f"view_{int(camid):05d}" / rgb_out_fname

        if l1[idx] > l1_thr:
            print(f"WARNING: Image {img_path} cannot be corrected, error is too high. ({l1[idx]:.3f})")
            continue

        if red_outlier_ratios[idx] > red_outlier_ratio_thr:
            print(f"WARNING: Image {img_path} cannot be corrected, red_outlier_ratio is too high. "
                    f"({red_outlier_ratios[idx]:.3f})")
            continue

        if l1[idx] < l1_corrected[idx]:
            print(f"WARNING: Couldnt reduce l1 error of {img_path}. Old l1: {l1[idx]:.3f}, "
                    f"New l1: {l1_corrected[idx]}. Copying GT image unchanged.")
            os.system(f"cp {img_path} {out_path}")
            continue

        img, alpha = (retry_f(pil_to_tensor, retry_f(Image.open, img_path)).to(device) / 255.).split([3, 1], dim=0)
        corrector = color_correctors[idx]
        img_corrected = img.reshape(3, -1)
        img_corrected = torch.cat((img_corrected, torch.ones_like(img_corrected[:1])), dim=0)  # (4, N)
        img_corrected = corrector @ img_corrected  # (3, N)
        img_corrected = img_corrected.view(*img.shape)

        # # visualizing image color correction
        # print(f"{camid} {sorted_l1s[idx]:.3f} {sorted_l1s_corrected[idx]:.3f}")
        # img = img.permute(1, 2, 0)
        # img[img.mean(dim=-1) >= specular_thr] = 0.
        # img = img.permute(2, 0, 1)
        # fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
        # axes[0].imshow(img.permute(1, 2, 0).cpu())
        # axes[1].imshow(img_corrected.permute(1, 2, 0).cpu())
        # plt.show()

        img_corrected = torch.cat((img_corrected, alpha), dim=0).cpu().clip(min=0, max=1.)
        retry_f(save_image, img_corrected, out_path)
        # except Exception as e:
        #     print("ERROR", e)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src", "-S")
    parser.add_argument("--ncams", default=-1, type=int)
    args = parser.parse_args()
    calibrate_colors(Path(args.src), ncams=args.ncams)