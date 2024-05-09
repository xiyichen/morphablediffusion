from math import log10, sqrt 
import cv2 
import numpy as np
from skimage.metrics import structural_similarity as SSIM
import lpips
import torch
import glob
import os, json
import mmpose.evaluation
from mmpose.evaluation.functional import keypoint_pck_accuracy
from torchmetrics.image.fid import FrechetInceptionDistance
import argparse
import dlib
from scipy.spatial import distance

def read_transparent_png(filename):
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:,:,3]
    rgb_channels = image_4channel[:,:,:3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return (final_image).astype(np.uint8)[:,:,::-1], white
    
def main():
    with open(os.path.join('./eval/facescape_input_target_views.json')) as f:
        test_metadata = json.load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['nvs', 'nes'])
    flags = parser.parse_args()
    
    pred_dir = f'./eval/facescape_bilinear_{flags.mode}_output'
    
    test_subjects = [str(i) for i in [122, 212] + list(range(326, 360))]
    if flags.mode == 'nes':
        test_exps = [6]
    else:
        test_exps = list(range(1, 21))
    test_exps = [str(i).zfill(2) for i in test_exps]
    
    fid = FrechetInceptionDistance().cuda()
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda() # closer to "traditional" perceptual loss, when used for optimization
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('./assets/dlib/shape_predictor_5_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('./assets/dlib/dlib_face_recognition_resnet_model_v1.dat')
    
    ssim = 0
    lpips_v = 0
    gt_kpts_all = []
    generated_kpts_all = []

    count = 0
    count_reid = 0
    fids_all = 0
    re_id = 0
    for subject_id in test_subjects:
        gt_all = []
        generated_all = []
        for exp_id in test_exps:
            print(subject_id, exp_id)
            if 'target_views' not in test_metadata[subject_id][exp_id]:
                continue
            if not os.path.isfile(f'./eval/kpts_gt/{subject_id}/{exp_id}/kpts.json'):
                print(f'Keypoints prediction ./eval/kpts_gt/{subject_id}/{exp_id}/kpts.json does not exist!')
                exit()
            else:
                with open(f'./eval/kpts_gt/{subject_id}/{exp_id}/kpts.json', 'r') as f:
                    kpts_gt = json.load(f)
            if not os.path.isfile(f'./eval/kpts_{flags.mode}/{subject_id}/{exp_id}/kpts.json'):
                print(f'Keypoints prediction ./eval/kpts_{flags.mode}/{subject_id}/{exp_id}/kpts.json does not exist!')
                exit()
            else:
                with open(f'./eval/kpts_{flags.mode}/{subject_id}/{exp_id}/kpts.json', 'r') as f:
                    kpts_generated = json.load(f)
                
            target_views = test_metadata[subject_id][exp_id]['target_views']
            generated_batch = cv2.imread(os.path.join(pred_dir, f'{subject_id}_{exp_id}.png'))[:,:,::-1][:,256:,:]
            
            for idx, target_view in enumerate(target_views):
                row_id = idx//16
                column_id = idx-16*(idx//16)
                generated_img = generated_batch[row_id*256:(row_id+1)*256,column_id*256:(column_id+1)*256]
                gt_img, gt_img_mask = read_transparent_png(os.path.join(flags.data_dir, f'{str(subject_id).zfill(3)}/{str(exp_id).zfill(2)}/view_{str(target_view).zfill(5)}/rgba_colorcalib.png'))
                generated_img[gt_img_mask[:,:,0]==255] = 255
                
                dets = detector(gt_img, 1)
                if len(dets) == 1:
                    shape_gt = sp(gt_img, dets[0])
                    face_descriptor_gt = facerec.compute_face_descriptor(gt_img, shape_gt)
                    face_descriptor_gt = np.array(face_descriptor_gt)
                    shape_generated = sp(generated_img, dets[0])
                    face_descriptor_generated = facerec.compute_face_descriptor(generated_img, shape_generated)
                    face_descriptor_generated = np.array(face_descriptor_generated)
                    d = distance.euclidean(face_descriptor_gt, face_descriptor_generated)
                    count_reid += 1
                    if d < 0.6:
                        re_id += 1
                
                ssim_one = SSIM(gt_img, generated_img, channel_axis=2, multichannel=True)
                gt_torch_one = torch.from_numpy(gt_img.copy()).permute(2,0,1).unsqueeze(0) / 255
                target_torch_one = torch.from_numpy(generated_img.copy()).permute(2,0,1).unsqueeze(0) / 255
                lpips_one = loss_fn_vgg(gt_torch_one.cuda(), target_torch_one.cuda(), normalize=True)[0][0][0][0].item()
                ssim += ssim_one
                lpips_v += lpips_one
                generated_all.append(target_torch_one*255)
                gt_all.append(gt_torch_one*255)
                if target_view in kpts_gt and target_view in kpts_generated:
                    gt_kpts = np.array(kpts_gt[target_view])[:,:2]
                    generated_kpts = np.array(kpts_generated[target_view])[:,:2]
                    gt_kpts_all.append(gt_kpts)
                    generated_kpts_all.append(generated_kpts)
                count += 1
        if len(gt_all) == 0:
            continue
        gt_all = torch.stack(gt_all).to(torch.uint8).cuda().squeeze(1)
        generated_all = torch.stack(generated_all).to(torch.uint8).cuda().squeeze(1)
        fid.update(gt_all, real=True)
        fid.update(generated_all, real=False)
        fid_val = fid.compute()
       
    gt_kpts_all = np.stack(gt_kpts_all)
    generated_kpts_all = np.stack(generated_kpts_all)
    # normalize keypoints with intercanthal distance between two eyes
    interocular = np.linalg.norm(gt_kpts_all[:, 39, :] - gt_kpts_all[:, 42, :], axis=1, keepdims=True)
    normalize = np.tile(interocular, [1, 2])
    pck = keypoint_pck_accuracy(generated_kpts_all, gt_kpts_all, mask=np.ones((len(gt_kpts_all), 68), dtype=bool), thr=0.2, norm_factor=normalize)[1]

    print(f"SSIM: {ssim / count}, LPIPS: {lpips_v / count}, FID: {fid_val}, PCK: {pck}, Re-ID: {re_id / count_reid}")

if __name__=="__main__":
    main()