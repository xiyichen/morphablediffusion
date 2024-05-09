# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv, glob
import mmengine
import numpy as np
from mmengine.logging import print_log
import glob

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='root of preprocessed FaceScape data')
    parser.add_argument(
        '--gt',
        action='store_true',
        help='predict keypoints for the ground truth views')
    parser.add_argument(
        '--save_projections',
        action='store_true',
        help='save images with projected 2D keypoints')
    parser.add_argument(
        '--pred_dir',
        type=str,
        help='root of generated_views')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument('--mode', type=str, default='nes', choices=['nvs', 'nes'],
                        help='nes: novel expression synthesis (only expression id 6); nvs: novel view synthesis (all expressions)')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    # assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)
    
    with open(os.path.join('./eval/facescape_input_target_views.json')) as f:
        test_metadata = json.load(f)
    
    if args.mode == 'nes':
        test_exps = [6]
    else:
        test_exps = list(range(1, 21))
    test_exps = [str(i).zfill(2) for i in test_exps]

    output_dir = './eval/kpts_' + ('gt' if args.gt else args.mode)
    for subject_id in test_metadata.keys():
        for exp_id in test_exps:
            os.makedirs(os.path.join(output_dir, f'{str(subject_id).zfill(3)}', f'{str(exp_id).zfill(2)}'), exist_ok=True)
        
            if os.path.isfile(os.path.join(output_dir, f'{str(subject_id).zfill(3)}', f'{str(exp_id).zfill(2)}', 'kpts.json')):
                continue
            data_dir = os.path.join(args.data_dir, f'{subject_id}/{exp_id}')
            views = glob.glob(os.path.join(data_dir, 'view_*'))
            if len(views) == 0:
                continue
            
            if not os.path.isfile(os.path.join(data_dir, 'cameras.json')):
                continue
            with open(os.path.join(data_dir, 'cameras.json'), 'r') as f:
                camera = json.load(f)
            target_views = test_metadata[subject_id][exp_id]['target_views']
            kpts_all = {}
            if not args.gt:
                generated_batch = cv2.imread(os.path.join(args.pred_dir, f'{subject_id}_{exp_id}.png'))[:,:,::-1][:,256:,:]
            
            for idx, target_view in enumerate(target_views):
                if abs(camera[target_view]['angles']['azimuth']) > 60 or abs(camera[target_view]['angles']['elevation']) > 30:
                    continue
                
                if not os.path.isfile(os.path.join(data_dir, f'view_{str(target_view).zfill(5)}/rgba_colorcalib.png')):
                    continue
                if args.gt:
                    img = cv2.imread(os.path.join(data_dir, f'view_{str(target_view).zfill(5)}/rgba_colorcalib.png'))[:,:,::-1]
                else:
                    row_id = idx//16
                    column_id = idx-16*(idx//16)
                    img = generated_batch[row_id*256:(row_id+1)*256,column_id*256:(column_id+1)*256]
                    
                pred_instances = process_one_image(args, img, detector,
                                                    pose_estimator, visualizer)
                pred_instances_list = split_instances(pred_instances)

                kpts = np.zeros((68, 3))
                kpts[:,:2] = np.array(pred_instances_list[0]['keypoints'])
                kpts[:,2] = np.array(pred_instances_list[0]['keypoint_scores'])
                kpts_all[str(target_view)] = kpts.tolist()
                
                
                if args.save_projections:
                    img_vis = visualizer.get_image()
                    mmcv.imwrite(mmcv.rgb2bgr(img_vis), os.path.join(output_dir, f'{str(subject_id).zfill(3)}', f'{str(exp_id).zfill(2)}', f'view_{str(target_view).zfill(5)}.png'))
            
            with open(os.path.join(output_dir, f'{str(subject_id).zfill(3)}', f'{str(exp_id).zfill(2)}', 'kpts.json'), 'w') as f:
                json.dump(
                    kpts_all,
                    f,
                    indent='\t')


if __name__ == '__main__':
    main()