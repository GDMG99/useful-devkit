import argparse
import pickle

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from useful import USEFUL


def get_images(dataset, img_sample_data_tokens):
    images_dict = {}
    for token in img_sample_data_tokens:
        sample_data_rec = dataset.get('sample_data', token)
        calib_rec = dataset.get('calibrated_sensor', sample_data_rec['calibrated_sensor_token'])
        sensor = sample_data_rec['channel']
        img_path = sample_data_rec['filename'].split('/')[-1]
        cam2img = calib_rec['camera_intrinsic']
        cam2img = np.array(cam2img).reshape((3, 3))
        trans = calib_rec['translation']
        rot = calib_rec['rotation']
        rot_q = Quaternion(rot)
        rot_m = rot_q.rotation_matrix
        H = np.eye(4)
        H[:3, :3] = rot_m
        H[:3, 3] = trans
        lidar2cam = H
        timestamp = sample_data_rec['timestamp']
        sample_data_token = token

        images_dict[sensor] = {
            'img_path': img_path,
            'cam2img': cam2img,
            'lidar2cam': lidar2cam,
            'timestamp': timestamp,
            'sample_data_token': sample_data_token,
            'height': sample_data_rec['height'],
            'width': sample_data_rec['width']
        }
    return images_dict


def get_instances(dataset, ann_tokens):
    map_categories = {
        'pedestrian': 'pedestrian',
        'animal': 'animal',
        'vehicle.car': 'car',
        'vehicle.motorcycle': 'motorcycle',
        'vehicle.bus': 'bus',
        'vehicle.truck': 'truck',
        'vehicle.construction': 'construction_vehicle',
        'sign': 'sign',
        'personal_mobility': 'personal_mobility',
        'bicycle': 'bicycle',
    }
    categories_dict = {
        'car': 0,
        'truck': 1,
        'sign': 2,
        'bus': 3,
        'construction_vehicle': 4,
        'bicycle': 5,
        'motorcycle': 6,
        'pedestrian': 7,
        'animal': 8,
        'personal_mobility': 9
    }
    inst_list = []
    for ann_token in ann_tokens:
        ann = dataset.get('sample_annotation', ann_token)

        cat_name = map_categories.get(ann['category_name'], '')
        bbox_label = categories_dict.get(cat_name, None)
        if bbox_label is None:
            continue

        center = np.array(ann['translation'])
        q = Quaternion(ann['rotation'])
        yaw = q.yaw_pitch_roll[0]

        # useful ann['size'] = [width, length, height]
        # mmdet3d LiDAR format: [x, y, z, l, w, h, yaw]  (l=size[1], w=size[0])
        w, l, h = ann['size'][0], ann['size'][1], ann['size'][2]

        bbox_3d = [float(center[0]), float(center[1]), float(center[2]),
                   float(w), float(l), float(h), float(yaw)]

        box_dict = {
            'bbox_label': bbox_label,
            'bbox_label_3d': bbox_label,
            'bbox_3d': bbox_3d,
            'bbox_3d_isvalid': True,
            'num_lidar_pts': ann['num_lidar_pts'],
            'num_radar_pts': 0,
            'velocity': [0.0, 0.0],
        }
        inst_list.append(box_dict)
    return inst_list


def get_lidar_points(dataset, sample_data_token):
    sample_data_rec = dataset.get('sample_data', sample_data_token)
    return {
        'lidar_path': sample_data_rec['filename'],
        'num_pts_feats': 5,
        'token': sample_data_token,
        'timestamp': sample_data_rec['timestamp']
    }


def get_radar_points(dataset, sample_data_dict):
    radar_dict = {}
    for sensor, token in sample_data_dict.items():
        if sensor in ['RADAR_RIGHT', 'RADAR_LEFT']:
            sample_data_rec = dataset.get('sample_data', token)
            rdr = {
                'lidar_path': sample_data_rec['filename'],
                'num_pts_feats': 16,
                'token': token,
                'timestamp': sample_data_rec['timestamp']
            }
            radar_dict[sensor] = rdr
    return radar_dict


def get_cam_instances(dataset, sample_data_dict):
    map_categories = {
        'pedestrian': 'pedestrian',
        'animal': 'animal',
        'vehicle.car': 'car',
        'vehicle.motorcycle': 'motorcycle',
        'vehicle.bus': 'bus',
        'vehicle.truck': 'truck',
        'vehicle.construction': 'construction_vehicle',
        'sign': 'sign',
        'personal_mobility': 'personal_mobility',
        'bicycle': 'bicycle',
    }
    categories_dict = {
        'car': 0,
        'truck': 1,
        'sign': 2,
        'bus': 3,
        'construction_vehicle': 4,
        'bicycle': 5,
        'motorcycle': 6,
        'pedestrian': 7,
        'animal': 8,
        'personal_mobility': 9
    }
    cam_instances = {}
    for sensor, token in sample_data_dict.items():
        if sensor in ['LIDAR', 'RADAR_LEFT', 'RADAR_RIGHT']:
            continue
        sample_data_rec = dataset.get('sample_data', token)
        ann_list = []
        for ann_token in sample_data_rec['anns_2d']:
            ann_rec = dataset.get('sample_annotation_2d', ann_token)
            cat_name = map_categories.get(ann_rec['category_name'], '')
            bbox_label = categories_dict.get(cat_name, None)
            if bbox_label is None:
                continue
            ann_dict = {
                'bbox': ann_rec['bbox'],
                'bbox_label': bbox_label,
                'center_2d': [
                    (ann_rec['bbox'][0] + ann_rec['bbox'][2]) / 2,
                    (ann_rec['bbox'][1] + ann_rec['bbox'][3]) / 2
                ]
            }
            ann_list.append(ann_dict)
        cam_instances[sensor] = ann_list
    return cam_instances


def get_sample_info(dataset, sample_token, sample_idx):
    sample_rec = dataset.get('sample', sample_token)
    img_tokens = [val for key, val in sample_rec['data'].items()
                  if key in ['WIDE_LEFT', 'WIDE_RIGHT', 'NARROW']]
    images_dict = get_images(dataset, img_tokens)
    lidar_dict = get_lidar_points(dataset, sample_rec['data']['LIDAR'])
    radar_dict = get_radar_points(dataset, sample_rec['data'])
    instances = get_instances(dataset, sample_rec['anns'])
    cam_instances = get_cam_instances(dataset, sample_rec['data'])

    return {
        'sample_idx': sample_idx,
        'token': sample_token,
        'timestamp': sample_rec['timestamp'],
        'images': images_dict,
        'lidar_points': lidar_dict,
        'radar_points': radar_dict,
        'instances': instances,
        'cam_instances': cam_instances,
    }


def get_infos_pkl(dataset, splits, metainfo, out_path):
    samples = dataset.get_sample_tokens_for_split(splits)
    data_list = []
    for i, sample in enumerate(tqdm(samples)):
        data_list.append(get_sample_info(dataset, sample, i))
    infos = {
        'metainfo': metainfo,
        'data_list': data_list
    }
    with open(out_path, 'wb') as f:
        pickle.dump(infos, f)
    print(f'Saved {out_path}')
    return infos


def main():
    parser = argparse.ArgumentParser(
        description='Generate mmdet3d-style .pkl info files from the USEFUL dataset.'
    )
    parser.add_argument('-v', '--version', required=True,
                        help='Dataset version (e.g. v0.7)')
    parser.add_argument('-p', '--dataroot', default='data/useful',
                        help='Path to the dataset root (default: data/useful)')
    parser.add_argument('-s', '--split', nargs='+', default=['test'],
                        help='Split(s) to process: train, val, test (default: test)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output .pkl file path')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose dataset loading output')
    args = parser.parse_args()

    usfl = USEFUL(version=args.version, dataroot=args.dataroot, verbose=args.verbose)

    metainfo = {
        'categories': {
            'car': 0,
            'truck': 1,
            'sign': 2,
            'bus': 3,
            'construction_vehicle': 4,
            'bicycle': 5,
            'motorcycle': 6,
            'pedestrian': 7,
            'animal': 8,
            'personal_mobility': 9
        },
        'dataset': 'useful',
        'version': '1.0',
        'info_version': '1.1'
    }

    get_infos_pkl(usfl, args.split, metainfo, args.output)


if __name__ == '__main__':
    main()
