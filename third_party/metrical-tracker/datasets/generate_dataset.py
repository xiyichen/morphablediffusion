import os
from abc import ABC
from glob import glob
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

from face_detector import FaceDetector
from image import crop_image_bbox, squarefiy, get_bbox
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

class GeneratorDataset(Dataset, ABC):
    def __init__(self, source, config):
        self.device = 'cuda:0'
        self.config = config
        self.source = Path(source)

        self.initialize()
        self.face_detector_mediapipe = FaceDetector('google')
        self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=self.device)
        
        self.detector = init_detector(
            '../mmpose/demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py', 'https://download.openmmlab.com/mmpose/mmdet_pretrained/yolo-x_8xb8-300e_coco-face_13274d7c.pth', device=self.device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
            
        self.pose_estimator = init_pose_estimator(
            '../mmpose/configs/face_2d_keypoint/topdown_heatmap/coco_wholebody_face/td-hm_hrnetv2-w18_dark-8xb32-60e_coco-wholebody-face-256x256.py',
            'https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_coco_wholebody_face_256x256_dark-3d9a334e_20210909.pth',
            device=self.device,
            cfg_options=dict(
                model=dict(test_cfg=dict(output_heatmaps=False))))

    def initialize(self):
        self.images = sorted(glob(f'{self.source}/source/*.jpg') + glob(f'{self.source}/source/*.png'))

    def process_face(self, image):
        # import pdb; pdb.set_trace()
        det_result = inference_detector(self.detector, image)
        pred_instance = det_result.pred_instances.cpu().numpy()
        # lmks, scores, detected_faces = self.face_detector.get_landmarks_from_image(image, return_landmark_score=True, return_bboxes=True)
        if pred_instance is None:
            lmks = None
        else:
            # for idx_, loc in enumerate(lmks):
            #     x = int(loc[0])
            #     y = int(loc[1])
            #     cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
            # cv2.imwrite('./debug.png', image)
            
            # import pdb; pdb.set_trace()
            pose_results = inference_topdown(self.pose_estimator, image, pred_instance.bboxes[0][:4].reshape(1,4))
            data_samples = merge_data_samples(pose_results)
            pred_instances = data_samples.get('pred_instances', None)
            pred_instances_list = split_instances(pred_instances)
            lmks = np.array(pred_instances_list[0]['keypoints']).astype(np.int32).astype(np.float32)
            
            for idx_, loc in enumerate(lmks):
                x = int(loc[0])
                y = int(loc[1])
                cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
        dense_lmks = self.face_detector_mediapipe.dense(image)
        if dense_lmks is not None:
            for idx_, loc in enumerate(dense_lmks):
                x = int(loc[0])
                y = int(loc[1])
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        
        cv2.imwrite('./debug.png', image[:,:,::-1])
        return lmks, dense_lmks
        # import pdb; pdb.set_trace()
        
        # lmks, scores, detected_faces = self.face_detector.get_landmarks_from_image(image, return_landmark_score=True, return_bboxes=True)
        # if detected_faces is None:
        #     lmks = None
        # else:
        #     lmks = lmks[0]
        #     for idx_, loc in enumerate(lmks):
        #         x = int(loc[0])
        #         y = int(loc[1])
        #         cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
        # dense_lmks = self.face_detector_mediapipe.dense(image)
        # if dense_lmks is not None:
        #     for idx_, loc in enumerate(dense_lmks):
        #         x = int(loc[0])
        #         y = int(loc[1])
        #         cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        # cv2.imwrite('./debug.png', image[:,:,::-1])
        # return lmks, dense_lmks

    def run(self):
        logger.info('Generating dataset...')
        bbox = None
        bbox_path = self.config.actor + "/bbox.pt"

        if os.path.exists(bbox_path):
            bbox = torch.load(bbox_path)

        for imagepath in tqdm(self.images):
            lmk_path = imagepath.replace('source', 'kpt').replace('png', 'npy').replace('jpg', 'npy')
            lmk_path_dense = imagepath.replace('source', 'kpt_dense').replace('png', 'npy').replace('jpg', 'npy')

            if not os.path.exists(lmk_path) or not os.path.exists(lmk_path_dense):
                image = cv2.imread(imagepath)
                h, w, c = image.shape

                if bbox is None and self.config.crop_image:
                    lmk, _ = self.process_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # estimate initial bbox
                    bbox = get_bbox(image, lmk, bb_scale=self.config.bbox_scale)
                    torch.save(bbox, bbox_path)

                if self.config.crop_image:
                    image = crop_image_bbox(image, bbox)
                    if self.config.image_size[0] == self.config.image_size[1]:
                        image = squarefiy(image, size=self.config.image_size[0])
                else:
                    image = cv2.resize(image, (self.config.image_size[1], self.config.image_size[0]), interpolation=cv2.INTER_CUBIC)

                lmk, dense_lmk = self.process_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if lmk is None:
                    logger.info(f'Empty face_alignment lmks for path: ' + imagepath)
                    lmk = np.zeros([68, 2])

                if dense_lmk is None:
                    logger.info(f'Empty mediapipe lmks for path: ' + imagepath)
                    dense_lmk = np.zeros([478, 2])

                Path(lmk_path).parent.mkdir(parents=True, exist_ok=True)
                Path(lmk_path_dense).parent.mkdir(parents=True, exist_ok=True)
                Path(imagepath.replace('source', 'images')).parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(imagepath.replace('source', 'images'), image)
                np.save(lmk_path_dense, dense_lmk)
                np.save(lmk_path, lmk)
