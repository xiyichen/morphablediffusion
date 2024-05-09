import argparse
import os, glob, json
from scipy.spatial.transform import Rotation as Rot
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    flags = parser.parse_args()
    test_subjects = [str(i) for i in [122, 212] + list(range(326, 360))]
    test_exps = list(range(1, 21))
    test_exps = [str(i).zfill(2) for i in test_exps]
    
    metadata = {}
    for subject_id in test_subjects:
        metadata[subject_id] = {}
        for expression_id in test_exps:
            metadata[subject_id][expression_id] = {}
            data_dir = os.path.join(flags.data_dir, f'{subject_id}/{expression_id}')
            views = glob.glob(os.path.join(data_dir, 'view_*'))
            if len(views) == 0:
                continue
            
            with open(os.path.join(data_dir, 'cameras.json'), 'r') as f:
                camera_dict = json.load(f)
            input_view_candidates = []
            for view in camera_dict:
                if camera_dict[view]['angles']['azimuth'] < 15 and camera_dict[view]['angles']['elevation'] < 15 and os.path.isdir(os.path.join(data_dir, f'view_{str(view).zfill(5)}')):
                    # exclude some upside down camera views
                    RT = np.array(camera_dict[view]['extrinsics'])
                    if abs(Rot.from_matrix(RT[:3,:3]).as_euler('xyz', True)[-1]) > 90:
                        continue
                    input_view_candidates.append((camera_dict[view]['angles']['azimuth'], view))
            input_view_candidates.sort()
            input_view = input_view_candidates[0][1]
            metadata[subject_id][expression_id]['input_view'] = input_view
            target_view_candidates = []
            for view in camera_dict:
                if camera_dict[view]['angles']['azimuth'] < 90 and os.path.isdir(os.path.join(data_dir, f'view_{str(view).zfill(5)}')):
                    RT = np.array(camera_dict[view]['extrinsics'])
                    if abs(Rot.from_matrix(RT[:3,:3]).as_euler('xyz', True)[-1]) > 90:
                        continue
                    target_view_candidates.append(view)
            metadata[subject_id][expression_id]['target_views'] = target_view_candidates
        
    with open("./eval/facescape_input_target_views.json", "w") as outfile: 
        json.dump(metadata, outfile)

if __name__=="__main__":
    main()