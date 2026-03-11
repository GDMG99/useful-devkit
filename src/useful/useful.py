import os
import sys
import json
import time
import numpy as np
import pathlib 
import cv2
import open3d as o3d
import folium
from IPython.display import display
import webbrowser
from pyquaternion import Quaternion
import tqdm

from datetime import datetime
from .utils.data_classes import Box, Box2D, LidarPointCloud, SWIRImage, RadarPointCloud, RGBImage, ThermalImage, PolarimetricImage

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("useful-devkit only supports Python version 3.")

class USEFUL:
    """
    Database class for useful for data management.
    """
    def __init__(self,
                 version: str = 'v0.0',
                 dataroot: str =  'data/useful',
                 verbose: bool = True,
                 table_names: list = ['log', 
                                      'calibrated_sensor',
                                      'sensor',
                                      'sample',
                                      'scene',
                                      'sample_data',
                                      'instance',
                                      'visibility',
                                      'category',
                                      'sample_annotation',
                                      'sample_annotation_2d',
                                      'ego_pose']) -> None:
        """
        Loads database
        :param version: Dataset version to look for
        :param dataroot: Path to the tables and data
        :param verbose: Wether you print status messages during load
        :param table_names: Tables to load
        """

        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.table_names = table_names

        assert os.path.exists(self.table_root), f'Database version not found: {self.table_root}'

        if verbose:
            start_time = time.time()
            print('Starting to load log...')
        self.log = self.__load_table__('log')
        if verbose:
            end_time = time.time()
            print(f'Finished loading log! ({end_time - start_time:2.3f} s)\n**********')
            print('Starting to load calibrated_sensor...')
            start_time = time.time()
        self.calibrated_sensor = self.__load_table__('calibrated_sensor')
        if verbose:
            end_time = time.time()
            print(f'Finished loading calibrated_sensor! ({end_time - start_time:2.3f} s)\n**********')
            print('Starting to load map...')
            start_time = time.time()
        if os.path.exists(os.path.join(self.table_root,'map.json')):
            self.map = self.__load_table__('map')
        else:
            self.map = None
            if verbose:
                print('map.json not found') 
        if verbose:
            end_time = time.time()
            print(f'Finished loading map! ({end_time - start_time:2.3f} s)\n**********')
            print('Starting to load sensor...')
            start_time = time.time()
        self.sensor = self.__load_table__('sensor')
        if verbose:
            end_time = time.time()
            print(f'Finished loading sensor! ({end_time - start_time:2.3f} s)\n**********')
            print('Starting to load sample...')
            start_time = time.time()
        self.sample = self.__load_table__('sample')
        if verbose:
            end_time = time.time()
            print(f'Finished loading sample! ({end_time - start_time:2.3f} s)\n**********')
            print('Starting to load scene...')
            start_time = time.time()
        self.scene = self.__load_table__('scene')
        if verbose:
            end_time = time.time()
            print(f'Finished loading scene! ({end_time - start_time:2.3f} s)\n**********')
            print('Starting to load sample_data...')
            start_time = time.time()
        self.sample_data = self.__load_table__('sample_data')
        if verbose:
            end_time = time.time()
            print(f'Finished loading sample_data! ({end_time - start_time:2.3f} s)\n**********')
            print('Starting to load sample_annotation...')
            start_time = time.time()
        if os.path.exists(os.path.join(self.table_root,'sample_annotation.json')):
            self.sample_annotation = self.__load_table__('sample_annotation')
        else:
            self.sample_annotation = None
            if verbose:
                print('sample_annotation.json not found') 
        if verbose:
            end_time = time.time()
            print(f'Finished loading sample_annotation! ({end_time - start_time:2.3f} s)\n**********')
            print('Starting to load sample_annotation_2d...')
            start_time = time.time()
        if os.path.exists(os.path.join(self.table_root,'sample_annotation_2d.json')):
            self.sample_annotation_2d = self.__load_table__('sample_annotation_2d')
        else:
            self.sample_annotation_2d = None
            if verbose:
                print('sample_annotation_2d.json not found') 
        if verbose:
            end_time = time.time()
            print(f'Finished loading sample_annotation_2d! ({end_time - start_time:2.3f} s)\n**********')
            print('Starting to load instance...')
            start_time = time.time()
        if os.path.exists(os.path.join(self.table_root,'instance.json')):
            self.instance = self.__load_table__('instance')
        else:
            self.instance = None
            if verbose:
                print('instance.json not found') 
        if verbose:
            end_time = time.time()
            print(f'Finished loading instance! ({end_time - start_time:2.3f} s)\n**********')
            print('Starting to load category...')
            start_time = time.time()
        if os.path.exists(os.path.join(self.table_root,'category.json')):
            self.category = self.__load_table__('category')
        else:
            self.category = None
            if verbose:
                print('category.json not found') 
        if verbose:
            end_time = time.time()
            print(f'Finished loading category! ({end_time - start_time:2.3f} s)\n**********')
            print('Starting to load visibility...')
            start_time = time.time()
        if os.path.exists(os.path.join(self.table_root,'visibility.json')):
            self.visibility = self.__load_table__('visibility')
        else:
            self.visibility = None
            if verbose:
                print('visibility.json not found') 
        if verbose:
            end_time = time.time()
            print(f'Finished loading visibility ({end_time - start_time:2.3f} s)\n**********')
            print('Starting to load ego_pose...')
            start_time = time.time()
        if os.path.exists(os.path.join(self.table_root,'ego_pose.json')):
            self.ego_pose = self.__load_table__('ego_pose')
        else:
            self.ego_pose = None
            if verbose:
                print('ego_pose.json not found')
        if verbose:
            end_time = time.time()
            print(f'Finished loading ego_pose ({end_time - start_time:2.3f} s)\n**********')
        self.__make_reverse_index__(verbose)
        
        self.explorer = UsefulExplorer(self)

    
    @property
    def table_root(self) -> str:
        """ 
        Returns the folder where the tables are stored for the relevant version.
        """
        return os.path.join(self.dataroot, self.version)
    
    def __load_table__(self,
                       table_name: str) -> dict:
        """
        Loads a table.
        :param table_name: Name of the json file.
        """
        with open(os.path.join(self.table_root,f'{table_name}.json')) as f:
            table = json.load(f)
        return table
    
    def __make_reverse_index__(self,
                               verbose: bool) -> None:
        """
        Adds redundant information to different elements of the dataset to ease data management.
        """
        start_time = time.time()
        if verbose:
            print("Starting reverse indexing...")
        
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()
            if getattr(self, table) is None:
                continue
            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member['token']] = ind
            
        # Decorate (adds short-cut) sample_annotation table with for category name.
        if self.sample_annotation is not None and self.instance is not None:
            for record in self.sample_annotation:
                inst = self.get('instance', record['instance_token'])
                record['category_name'] = self.get('category', inst['category_token'])['name']
        if self.sample_annotation_2d is not None and self.instance is not None:
            for record in self.sample_annotation_2d:
                inst = self.get('instance', record['instance_token'])
                record['category_name'] = self.get('category', inst['category_token'])['name']
            
        # Adds short-cut to sample_data with sensor information
        for record in self.sample_data:
            cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self.get('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']
        
        # Reverse-index samples with sample_data and annotations
        for record in self.sample:
            record['data'] = {}
            record['anns'] = []
        for record in self.sample_data:
            try:
                sample_record = self.get('sample', record['sample_token'])
                sample_record['data'][record['channel']] = record['token']
            except KeyError:
                continue
        
        if self.sample_annotation is not None:
            for ann_record in self.sample_annotation:
                try:
                    sample_record = self.get('sample', ann_record['sample_token'])
                    sample_record['anns'].append(ann_record['token'])
                except KeyError:
                    continue
        
        # Reverse-index sample_data with 2d annotations
        for record in self.sample_data:
            if record['channel'] in ['LIDAR', 'RADAR_LEFT', 'RADAR_RIGHT']:
                continue
            record['anns_2d'] = []
        
        if self.sample_annotation_2d is not None:
            for ann_2d_record in self.sample_annotation_2d:
                try:
                    sample_data_record = self.get('sample_data', ann_2d_record['sample_data_token'])
                    sample_data_record['anns_2d'].append(ann_2d_record['token'])
                except KeyError:
                    continue
            
        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n**********".format(time.time() - start_time))

    @staticmethod
    def generate_token():
        """
        Generates random token.
        """
        token = ''
        for _ in range(32):
            token += "{:x}".format(np.random.randint(0, 16))
        return token
    
    def get(self,
            table_name: str,
            token: str) -> dict:
        """
        Gets a specific record of a table given a token.
        :param table_name: Desired record type.
        :param token: Desired token.
        :return: Desired record.
        """
        assert table_name in self.table_names, f"Table {table_name} not found in version {self.version}"
        return getattr(self, table_name)[self.getind(table_name, token)]
    
    def getind(self,
               table_name: str,
               token: str) -> int:
        """
        Returns the index of the record in a table.
        :param table_name: Desired record type.
        :param token: Desired token.
        :return: index
        """
        try: 
            return self._token2ind[table_name][token]
        except KeyError:
            print(f'KeyError: Token {token} not found in {table_name}')
            raise
        
    def get_sample_files(self,
                         sample_token: str) -> dict:
        '''
        Returns a dictionary with the filenames of each sample_data in the sample.
        Calls explorer method.
        '''
        return self.explorer.get_sample_files(sample_token)
    
    def get_sample_tokens_in_scene(self,
                                   scene_token: str) -> list:
        '''
        Returns a list with the tokens of each sample_data in the sample.
        '''
        scene_record = self.get('scene', scene_token)
        sample_token_list = []
        sample_record = self.get('sample', scene_record['first_sample_token'])
        for i in range(scene_record['nbr_samples']):
            sample_token_list.append(sample_record['token'])
            if sample_record['next'] != '':
                sample_record = self.get('sample', sample_record['next'])
            else:
                break
        return sample_token_list
    
    def get_annotations_for_instance(self,
                                     instance_token: str) -> list:
        instance_record = self.get('instance', instance_token)
        annotations_dict = {}
        if instance_record['nbr_annotations'] > 0:
            sample_anns = []
            curr_ann = instance_record['first_annotation_token']
            while curr_ann != '' and curr_ann != None:
                ann_record = self.get('sample_annotation', curr_ann)
                sample_anns.append(ann_record['token'])
                curr_ann = ann_record['next']
            annotations_dict['LIDAR'] = sample_anns
        for sensor in instance_record['annotations_2d']:
            if instance_record['annotations_2d'][sensor]['nbr_annotations'] > 0:
                sensor_anns = []
                curr_ann = instance_record['annotations_2d'][sensor]['first_annotation_token']
                while curr_ann != '' and curr_ann != None:
                    ann_record = self.get('sample_annotation_2d', curr_ann)
                    sensor_anns.append(ann_record['token'])
                    curr_ann = ann_record['next']
                annotations_dict[sensor] = sensor_anns
        return annotations_dict
    
    def get_annotations_in_scene(self,
                                 scene_token: str) -> dict:
        '''
        Returns a dictionary with the annotations in a scene.
        {sample_token: {instance_token: {SENSOR: sample_annotation_token}}}
        '''

        annotation_dict = {}
        sample_token_list = self.get_sample_tokens_in_scene(scene_token)
        for i in range(len(sample_token_list)):
            sample_record = self.get('sample', sample_token_list[i])
            annotation_dict[sample_record['token']] = {}
            for j in range(len(sample_record['anns'])):
                ann_record = self.get('sample_annotation', sample_record['anns'][j])
                if ann_record['instance_token'] not in annotation_dict[sample_record['token']].keys():
                    annotation_dict[sample_record['token']][ann_record['instance_token']] = {}
                annotation_dict[sample_record['token']][ann_record['instance_token']]['LIDAR'] = ann_record['token']
            for channel in sample_record['data'].keys():
                if channel in ['LIDAR', 'RADAR_LEFT', 'RADAR_RIGHT']:
                    continue
                sd_record = self.get('sample_data', sample_record['data'][channel])
                for k in range(len(sd_record['anns_2d'])):
                    ann_2d_record = self.get('sample_annotation_2d', sd_record['anns_2d'][k])
                    if ann_2d_record['instance_token'] not in annotation_dict[sample_record['token']].keys():
                        annotation_dict[sample_record['token']][ann_2d_record['instance_token']] = {}
                    annotation_dict[sample_record['token']][ann_2d_record['instance_token']][channel] = ann_2d_record['token']
        return annotation_dict
    
    def get_scene_tokens_for_split(self,
                                   split: list = ['train', 'test', 'val']) -> list:
        return [sc['token'] for sc in self.scene if sc['split'] in split]
    
    def get_sample_tokens_for_split(self,
                                   split: list = ['train', 'test', 'val']) -> list:
        scene_tokens = self.get_scene_tokens_for_split(split)
        sample_list = []
        for token in scene_tokens:
            sample_tokens_in_scene = self.get_sample_tokens_in_scene(token)
            sample_list.extend(sample_tokens_in_scene)
        return sample_list
        
        
    def get_ego_trajectory(self,
                           sample_token: str,
                           num_samples: int,
                           forward: bool = True) -> list:
        '''
        Returns ego-motion data for a sequence of samples expressed in the local body
        frame of the reference sample (sample_token).

        This is the inverse of PointCloud.local_to_geodetic: positions, velocities and
        orientations from lat/lon/heading are converted to the body frame of sample 0.

        :param sample_token: Reference sample token (defines the origin frame).
        :param num_samples: Number of additional samples to traverse beyond the reference.
        :param forward: If True, traverse via sample['next']; if False, via sample['prev'].
        :return: List of up to num_samples+1 dicts, one per sample:
            {
              'token': str,        # sample token
              'x': float,          # forward offset in ref body frame (m)
              'y': float,          # lateral offset in ref body frame (m)
              'z': float,          # vertical offset in ref body frame (m)
              'vx': float,         # forward velocity in current body frame (m/s)
              'vy': float,         # lateral velocity in current body frame (m/s)
              'R': np.ndarray,     # 3x3 rotation: current body frame -> ref body frame
            }
        The first entry always has x=y=z=0 and R=identity.
        '''
        try:
            from pyproj import CRS, Transformer
        except ImportError as e:
            raise ImportError(
                "The 'pyproj' package is required for get_ego_trajectory. "
                "Install it with `pip install pyproj`."
            ) from e

        geodetic_crs = CRS.from_epsg(4979)
        ecef_crs = CRS.from_epsg(4978)
        geo_to_ecef = Transformer.from_crs(geodetic_crs, ecef_crs, always_xy=True)

        def enu_to_ecef_matrix(lat, lon):
            lat_r = np.radians(lat)
            lon_r = np.radians(lon)
            return np.array([
                [-np.sin(lon_r), -np.sin(lat_r) * np.cos(lon_r),  np.cos(lat_r) * np.cos(lon_r)],
                [ np.cos(lon_r), -np.sin(lat_r) * np.sin(lon_r),  np.cos(lat_r) * np.sin(lon_r)],
                [ 0,              np.cos(lat_r),                   np.sin(lat_r)]
            ])

        def make_R_heading(heading_deg):
            h = -(np.radians(heading_deg) - np.pi / 2)
            return np.array([
                [ np.cos(h), -np.sin(h), 0],
                [ np.sin(h),  np.cos(h), 0],
                [ 0,          0,         1]
            ])

        ref_record = self.get('sample', sample_token)
        ref_ins = self.get('ego_pose', ref_record['ego_pose_token'])['INS']
        ref_lon = float(ref_ins['lon_deg'])
        ref_lat = float(ref_ins['lat_deg'])
        ref_alt = float(ref_ins['alt_m'])
        ref_ecef = np.array(geo_to_ecef.transform(ref_lon, ref_lat, ref_alt))
        R_enu_ecef = enu_to_ecef_matrix(ref_lat, ref_lon)
        R_heading_ref = make_R_heading(float(ref_ins['heading_deg']))

        results = []
        current_token = sample_token
        for _ in range(num_samples + 1):
            rec = self.get('sample', current_token)
            ins = self.get('ego_pose', rec['ego_pose_token'])['INS']

            lon = float(ins['lon_deg'])
            lat = float(ins['lat_deg'])
            alt = float(ins['alt_m'])

            ecef_i = np.array(geo_to_ecef.transform(lon, lat, alt))
            enu = R_enu_ecef.T @ (ecef_i - ref_ecef)
            local = R_heading_ref.T @ enu

            R_heading_i = make_R_heading(float(ins['heading_deg']))
            v_enu = np.array([float(ins['velocity_east_mps']),
                              float(ins['velocity_north_mps']),
                              -float(ins['velocity_down_mps'])])
            v_local = R_heading_ref.T @ v_enu

            R_rel = R_heading_ref.T @ R_heading_i

            results.append({
                'token': current_token,
                'x': float(local[0]),
                'y': float(local[1]),
                'z': float(local[2]),
                'vx': float(v_local[0]),
                'vy': float(v_local[1]),
                'R': R_rel
            })

            next_key = 'next' if forward else 'prev'
            next_tok = rec.get(next_key, '')
            if not next_tok:
                break
            current_token = next_tok

        return results

    def get_scene_files(self,
                        scene_token: str) -> dict:
        '''
        Returns a dictionary with the filenames of each sample_data in the scene.
        Calls explorer method.
        '''
        return self.explorer.get_scene_files(scene_token)
    
    def get_scenes_for_log(self,
                           log_token: str) -> list:
        '''
        Returns a list with the scene tokens for a log.
        Calls explorer method.
        '''
        return self.explorer.get_scenes_for_log(log_token)
    
    def get_matrices(self,
                     calibrated_sensor_token: str) -> dict:
        '''
        Returns the camera intrinsic matrix and the extrinsic matrix of a calibrated_sensor.
        Calls explorer method.
        '''
        return self.explorer.get_matrices(calibrated_sensor_token)
    
    def get_categories(self):
        '''
        Returns a dictionary with the categories of the dataset.
        Calls explorer method.
        '''
        return self.explorer.get_categories()

    
    def list_scenes(self) -> None:
        '''
        Calls explorer method.
        '''
        self.explorer.list_scenes()

    def list_sample(self,
                    sample_token: str) -> None:
        '''
        Calls explorer method.
        '''
        self.explorer.list_sample(sample_token)
    
    def render_sample_data(self,
                           sample_data_token: str,
                           show: bool = True,
                           with_anns: bool = True,
                           with_lidar: bool = False,
                           intensity: bool = False,
                           max_dist: float = np.inf,
                           loop: float = None,
                           verbose: bool = False,
                           width: int = 640,
                           height: int = 480,
                           lidar_format: str = 'USEFUL'):
        '''
        Calls explorer method.
        '''
        sample_data = self.explorer.render_sample_data(sample_data_token,
                                                       with_lidar = with_lidar,
                                                       intensity = intensity,
                                                       max_dist = max_dist,
                                                       loop = loop,
                                                       with_anns = with_anns,
                                                       verbose = verbose,
                                                       width = width,
                                                       height = height,
                                                       show = show,
                                                       lidar_format = lidar_format)
        return sample_data


    def render_sample(self,
                      sample_token: str,
                      with_anns: bool = False,
                      list_instances: list = [],
                      with_instance: bool = False,
                      with_lidar: bool = False,
                      loop: float = None,
                      intensity: bool = False,
                      filter_category: list = [],
                      show: bool = True,
                      show_pcd: bool = True,
                      max_dist: float = np.inf,
                      verbose: bool = True,
                      width: int = 640,
                      height: int = 480,
                      polMode: str = 'RGB',
                      canvas_shape: tuple = (2,3),
                      canvas_order: list = ['WIDE_LEFT',
                                            'NARROW',
                                            'WIDE_RIGHT',
                                            'LWIR',
                                            'POLARIMETRIC',
                                            'SWIR'],
                      lidar_format: str = 'USEFUL'):
        
        canvas, geometries = self.explorer.render_sample(sample_token=sample_token,
                                    with_anns = with_anns,
                                    list_instances = list_instances,
                                    with_instance = with_instance,
                                    with_lidar = with_lidar,
                                    loop = loop,
                                    intensity = intensity,
                                    filter_category = filter_category,
                                    show=show,
                                    show_pcd = show_pcd,
                                    max_dist = max_dist,
                                    verbose = verbose,
                                    width = width,
                                    height = height,
                                    canvas_shape=canvas_shape,
                                    canvas_order=canvas_order,
                                    lidar_format = lidar_format,
                                    polMode=polMode)
        return canvas, geometries

    def render_sample_bev(self,
                          sample_token: str,
                          width: int = 1800,
                          height: int = 800,
                          with_anns: bool = True,
                          bev_max_range: float = 150.0,
                          bev_resolution: float = 0.1,
                          intensity: bool = False,
                          show: bool = True) -> np.ndarray:
        '''
        Renders the BEV + 6-camera composite for a single sample.
        Calls explorer method.
        '''
        return self.explorer.render_sample_bev(sample_token=sample_token,
                                               width=width,
                                               height=height,
                                               with_anns=with_anns,
                                               bev_max_range=bev_max_range,
                                               bev_resolution=bev_resolution,
                                               intensity=intensity,
                                               show=show)

    def render_lidar_bev(self,
                         sample_token: str,
                         bev_max_range: float = 150.0,
                         bev_resolution: float = 0.1,
                         with_anns: bool = True,
                         intensity: bool = False,
                         show: bool = True) -> np.ndarray:
        '''
        Renders the BEV of the LiDAR point cloud for a single sample.
        Calls explorer method.
        '''
        return self.explorer.render_lidar_bev(sample_token=sample_token,
                                              bev_max_range=bev_max_range,
                                              bev_resolution=bev_resolution,
                                              with_anns=with_anns,
                                              intensity=intensity,
                                              show=show)

    def get_sample_data_path(self,
                             sample_data_token: str) -> str:
        '''
        Returns the path to a data sample.
        '''
        sd_record = self.get('sample_data', sample_data_token)
        return os.path.join(self.dataroot, 'samples', sd_record['channel'], sd_record['filename']) 

    
    def get_sample_data(self,
                        sample_data_token: str,
                        selected_anntokens: list = None) -> tuple:
        '''
        Returns: data_path, boxes, camera_intrinsic, extrinsic, distortion_vector
        '''
        sd_record = self.get('sample_data', sample_data_token)
        cs_record = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = self.get('sensor', cs_record['sensor_token'])

        data_path = self.get_sample_data_path(sample_data_token)

        cam2img, lidar2cam, dist_coeffs = self.get_matrices(cs_record['token'])
        # Boxes have to be in the desired sensor's system of coordinates
        boxes = []
        if selected_anntokens == False:
            boxes = []
        else:
            if sensor_record['modality'] in ['rgb', 'swir', 'thermal', 'polarimetric']:
                if selected_anntokens is not None:
                    for ann_token in selected_anntokens:
                        boxes.append(self.get_box_2d(ann_token))
                else:
                    boxes = self.get_boxes_2d(sample_data_token)
            elif sensor_record['modality'] in ['lidar']:
                if selected_anntokens is not None:
                    for ann_token in selected_anntokens:
                        boxes.append(self.get_box(ann_token))
                else:
                    boxes = self.get_boxes(sd_record['sample_token'])
                    
        return data_path, boxes, cam2img, lidar2cam, dist_coeffs

    def get_box(self,
                sample_annotation_token: str) -> Box:
        """
        Instantiates a Box class from a sample annotation token.
        :param sample_annotation_token: Unique sample_annotation identifier.
        :return box: Box object.
        """
        sa_record = self.get('sample_annotation', sample_annotation_token)
        box = Box(center=sa_record['translation'],
                  size=sa_record['size'],
                  orientation=Quaternion(sa_record['rotation']),
                  name=sa_record['category_name'],
                  label=self.get_categories()[sa_record['category_name']],
                  instance_token=sa_record['instance_token'],
                  token=sa_record['token'])
        return box
    
    def get_boxes(self,
                  sample_token: str) -> list:
        """
        Instantiates a Box class from a sample annotation token.
        :param sample_annotation_token: Unique sample_annotation identifier.
        :return box: Box object.
        """
        sd_record = self.get('sample', sample_token)
        boxes = []
        for i in range(len(sd_record['anns'])):
            boxes.append(self.get_box(sd_record['anns'][i]))
        return boxes
    
    def get_box_2d(self,
                   sample_annotation_2d_token: str) -> Box:
        """
        Instantiates a Box class from a sample annotation token.
        :param sample_annotation_token: Unique sample_annotation identifier.
        :return box: Box object.
        """
        sa_record = self.get('sample_annotation_2d', sample_annotation_2d_token)
        if 'visibility_token' in sa_record:
            box = Box2D(bbox=sa_record['bbox'],
                    name=sa_record['category_name'],
                    label= self.get_categories()[sa_record['category_name']],
                    token=sa_record['token'],
                    instance_token=sa_record['instance_token'],
                    visibility = sa_record['visibility_token'])
        else:
            box = Box2D(bbox=sa_record['bbox'],
                        name=sa_record['category_name'],
                        label= self.get_categories()[sa_record['category_name']],
                        token=sa_record['token'],
                        instance_token=sa_record['instance_token'])
        return box
    
    def get_boxes_2d(self,
                     sample_data_token: str) -> list:
        """
        Instantiates a Box class from a sample annotation token.
        :param sample_annotation_token: Unique sample_annotation identifier.
        :return box: Box object.
        """
        sd_record = self.get('sample_data', sample_data_token)
        boxes = []
        for i in range(len(sd_record['anns_2d'])):
            boxes.append(self.get_box_2d(sd_record['anns_2d'][i]))
        return boxes
    
    def generate_version(self):
        '''
        This method will generate a sub-dataset based on some given filters such as elements in the description of scenes, scenes tokens, scenes where 
        certain annotation categories are present, certain sensors, etc.
        '''
        print(f'This method is still not implemented, come back later')
        pass
    def render_map(self,
                   scene_token: str,
                   inline: bool = False,
                   gps: bool = True,
                   ins: bool = True,
                   browser: bool = True,
                   save_path: str = None):
        return self.explorer.render_map(scene_token=scene_token,
                                        inline=inline, browser=browser,
                                        gps=gps,
                                        ins=ins,
                                        save_path=save_path)
    
    def render_scene(self,
                     scene_token: str,
                     save_path: str,
                     fps: int = 7,
                     width: int = 1800,
                     height: int = 800,
                     with_anns: bool = True,
                     bev_max_range: float = 150.0,
                     bev_resolution: float = 0.1,
                     intensity: bool = False):
        return self.explorer.render_scene(scene_token=scene_token,
                                          save_path=save_path,
                                          fps=fps,
                                          width=width,
                                          height=height,
                                          with_anns=with_anns,
                                          bev_max_range=bev_max_range,
                                          bev_resolution=bev_resolution,
                                          intensity=intensity)
class UsefulExplorer:
    """
    Helper class to list and visualize Useful data. These are meant to serve as tutorials and templates for working with the data.
    """
    def __init__(self,
                 useful: USEFUL) -> None:
        self.useful = useful
    def list_scenes(self) -> None:
        """ Lists all scenes with some metadata. """
        # List of tuples (timestamp, useful.scene) of the first sample in each scene of useful.scene
        recs = [(self.useful.get('sample', record['first_sample_token'])['timestamp'], record) for record in self.useful.scene]

        for start_time, record in sorted(recs):
            start_time = start_time / 1e6
            length_time = self.useful.get('sample', record['last_sample_token'])['timestamp'] / 1e6 - start_time
            location = self.useful.get('log', record['log_token'])['location']
            desc = record['name'] + ', ' + record['description']
            if len(desc) > 55:
                desc = desc[:51] + "..."
            if len(location) > 18:
                location = location[:18]
            print(f'{desc:16}\nToken: {record["token"]}\nDate: {datetime.utcfromtimestamp(start_time).strftime("%y-%m-%d %H:%M:%S")}\nDuration: {length_time:4.0f}s\nLocation: {location}\n**********')

    def get_sample_files(self,
                         sample_token: str) -> dict:
        """
        Returns a dictionary with the filenames of each sample_data in the sample.
        """
        record = self.useful.get('sample', sample_token)
        files = {}
        for i in record['data'].keys():
            files[f'{i}'] = os.path.join(self.useful.dataroot, 'samples', i, self.useful.get('sample_data', record['data'][i])['filename'] )
        return files
    
    def get_scene_files(self,
                        scene_token: str) -> dict:
        """
        Returns a dictionary with the filenames of each sample_data in the sample.
        """
        scene_record = self.useful.get('scene', scene_token)
        files = {}
        token = scene_record['first_sample_token']
        for i in range(scene_record['nbr_samples']):
            sample_record = self.useful.get('sample', token)
            sample_dict = self.get_sample_files(sample_record['token'])
            for key in sample_dict.keys():
                if key in files.keys():
                    files[f'{key}'].append(sample_dict[key])
                else:
                    files[f'{key}'] = [sample_dict[key]]
            token = sample_record['next']
            if token == '':
                break
        return files
    def get_scenes_for_log(self,
                           log_token: str) -> list:
        scenes_list = []
        for scene in self.useful.scene:
            if scene['log_token'] == log_token:
                scenes_list.append(scene['token'])

        return scenes_list
    def list_sample(self,
                     sample_token: str) -> None:
        """
        Prints sample_data and sample_annotation related to sample.
        """

        sample_record = self.useful.get('sample', sample_token)
        print(f"Sample: {sample_record['token']}")
        for sd_token in sample_record['data'].values():
            sd_record = self.useful.get('sample_data', sd_token)
            print(f'- sample_data_token: {sd_token}')
            print(f'- modality: {sd_record["sensor_modality"]}')
            print(f'- channel: {sd_record["channel"]}')
            print(f'- time_diff_us: {sd_record["time_diff_us"]}')
            print('--')

        if 'anns' in sample_record:
            print('* Annotations in sample:')
            for ann_token in sample_record['anns']:
                ann_record = self.useful.get('sample_annotation', ann_token)
                print(f'* sample_annotation_token: {ann_record["token"]}')
                print(f'* category: {ann_record["category_name"]}')
                print('**')

    
    def render_sample_data(self,
                           sample_data_token: str,
                           with_anns: bool = True,
                           with_lidar: bool = False,
                           intensity: bool = False,
                           max_dist:float = np.inf,
                           loop: float = None,
                           verbose: bool = True,
                           width: int = 640,
                           height: int = 480,
                           show: bool = True,
                           lidar_format: str = 'USEFUL',
                           vis: o3d.visualization.Visualizer = None):
        '''
        '''
        # Get sensor modality
        sd_record = self.useful.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            data_path, boxes, camera_intrinsic, extrinsic, distortion_vector = self.useful.get_sample_data(sample_data_token)
            pointcloud = LidarPointCloud(data_path=data_path, format= lidar_format)
            pointcloud.filter_distance(max=max_dist)
            if verbose:
                print(pointcloud)
            if with_anns:
                boxes_list = [i.get_OBB() for i in boxes]
            if show:
                pointcloud.render(vis=vis, other_geometries=boxes_list, intensity=intensity, loop=loop)
            geometries = []
            geometries.append(pointcloud.point_cloud)
            if with_anns:
                geometries.extend(boxes_list)
            return geometries

        elif sensor_modality in ['swir', 'rgb', 'thermal', 'polarimetric']:
            data_path, boxes, camera_intrinsic, extrinsic, distortion_vector = self.useful.get_sample_data(sample_data_token)

            if not with_anns:
                boxes = []
            if sensor_modality == 'swir':
                image = SWIRImage(data_path)
                if verbose:
                    print(image)
                img = image.render(width=width, height=height, boxes=boxes, bbox_2d=True, show=False)
            elif sensor_modality == 'rgb':
                image = RGBImage(data_path=data_path)
                if verbose:
                    print(image)
                img = image.render(width=width, height=height, boxes=boxes, bbox_2d=True, show=False)

            elif sensor_modality == 'thermal':
                image = ThermalImage(data_path)
                if verbose:
                    print(image)
                img = image.render(width=width, height=height, boxes=boxes, bbox_2d=True, show=False)
            elif sensor_modality == 'polarimetric':
                # image = PolarimetricImage(data_path) TODO
                image = RGBImage(data_path=data_path)
                if verbose:
                    print(image)
                img = image.render(width=width, height=height, boxes=boxes, bbox_2d=True, show=False)
            if with_lidar:
                lidar_path = self.useful.get_sample_data_path(self.useful.get('sample', sd_record['sample_token'])['data']['LIDAR'])
                lidar = LidarPointCloud(data_path=lidar_path, format='USEFUL')
                lidar.filter_distance(max=max_dist)
                pixels = lidar.project_points_to_image(camera_intrinsics=camera_intrinsic,
                                                       extrinsic=extrinsic,
                                                       distortion_vector=distortion_vector)
                if intensity:
                    colors = lidar.get_color_for_intensity(max=2000)
                else:
                    colors = lidar.get_color_for_distance(color_palette='jet', loop=loop)
                for j in range(len(pixels)):
                    img = cv2.circle(img,
                                     (int(pixels[j, 0]), int(pixels[j, 1])),
                                     radius=1,
                                     color=(colors[j, 2], colors[j, 1], colors[j, 0]),
                                     thickness=-1)
            if show:
                cv2.imshow(f'{sample_data_token}', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return img

    def render_sample(self,
                      sample_token: str,
                      with_anns: bool = True,
                      with_instance: bool = False,
                      list_instances: list = [],
                      with_lidar: bool = False,
                      loop: float = None,
                      intensity: bool = False,
                      filter_category: list = [],
                      show_pcd: bool = True,
                      max_dist: float = np.inf,
                      verbose: bool = True,
                      width: int = 640,
                      height: int = 480,
                      cell_size: tuple = None,
                      canvas_shape: tuple = None,
                      canvas_order: list = None,
                      show: bool = True,
                      polMode: str = 'RGB',
                      window_name: str = f'Sample',
                      lidar_format: str = 'USEFUL') -> tuple:
        
        if verbose:
            self.useful.list_sample(sample_token)
        s_record = self.useful.get('sample', sample_token)
        records_2d = []
        imgs = {}
        record_lidar = []
        record_radar = []
        intrinsics_dir = {}
        extrinsic_dir = {}
        radar_extrinsics = []
        distortion_vector_dir = {}
        boxes_dir = {}
        channels_in_sample = list(s_record['data'].keys())
        for i in range(len(channels_in_sample)):
            sd_record = self.useful.get('sample_data', s_record['data'][channels_in_sample[i]])
            data_path, boxes, camera_intrinsics, extrinsic, distortion_vector = self.useful.get_sample_data(sd_record['token'])
            if sd_record['sensor_modality'] in ['rgb', 'swir', 'thermal', 'polarimetric']:
                records_2d.append(sd_record)

                if sd_record['sensor_modality'] == 'rgb':
                    imgs[sd_record['channel']] = RGBImage(data_path=data_path)
                    intrinsics_dir[sd_record['channel']] = camera_intrinsics
                    extrinsic_dir[sd_record['channel']] = extrinsic
                    distortion_vector_dir[sd_record['channel']] = distortion_vector
                    boxes_dir[sd_record['channel']] = boxes
                elif sd_record['sensor_modality'] == 'swir':
                    imgs[sd_record['channel']] = SWIRImage(data_path)
                    intrinsics_dir[sd_record['channel']] = camera_intrinsics
                    extrinsic_dir[sd_record['channel']] = extrinsic
                    distortion_vector_dir[sd_record['channel']] = distortion_vector
                    boxes_dir[sd_record['channel']] = boxes
                elif sd_record['sensor_modality'] == 'thermal':
                    imgs[sd_record['channel']] = ThermalImage(data_path)
                    intrinsics_dir[sd_record['channel']] = camera_intrinsics
                    extrinsic_dir[sd_record['channel']] = extrinsic
                    distortion_vector_dir[sd_record['channel']] = distortion_vector
                    boxes_dir[sd_record['channel']] = boxes
                elif sd_record['sensor_modality'] == 'polarimetric':
                    pol = PolarimetricImage(data_path).getMode(polMode)[:,:,[2,1,0]]
                    pol = (pol / np.max(pol)) * 255
                    imgs[sd_record['channel']] = RGBImage(img = pol.astype(np.uint8))
                    intrinsics_dir[sd_record['channel']] = camera_intrinsics
                    extrinsic_dir[sd_record['channel']] = extrinsic
                    distortion_vector_dir[sd_record['channel']] = distortion_vector
                    boxes_dir[sd_record['channel']] = boxes
            elif sd_record['sensor_modality'] in ['lidar']:
                record_lidar.append(LidarPointCloud(data_path=data_path, format=lidar_format))
                record_lidar[0].filter_distance(max = max_dist)
                boxes_dir[sd_record['channel']] = boxes
            elif sd_record['sensor_modality'] in ['radar']:
                radar_extrinsics.append(extrinsic) 
                record_radar.append(RadarPointCloud(data_path=data_path))

        if cell_size is None and len(records_2d) != 0:
            cell_height = records_2d[0]['height']
            cell_width = records_2d[0]['width']
        
        if canvas_shape == None:
            num_rows = 1
            num_cols = len(imgs) // 2 + len(imgs) % 2
        else:
            num_rows, num_cols = canvas_shape

        canvas_height = num_rows * cell_height
        canvas_width = num_cols * cell_width
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        if type(with_lidar) == bool:
            with_lidar = [with_lidar for _ in range(num_rows * num_cols)]
        if type(with_anns) == bool:
            with_anns = [with_anns for _ in range(num_rows * num_cols)]

        count_imgs = 0
        for i in range( num_cols * num_rows):
            if type(canvas_order) == list and canvas_order[i] is not None:
                img = imgs[canvas_order[i]].render(show = False)
                
                row = i // num_cols
                col = i % num_cols
                x_start = col * cell_width
                x_end = x_start + cell_width
                y_start = row * cell_height
                y_end = y_start + cell_height

                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                resized = cv2.resize(img, (cell_width, cell_height))
                if with_lidar[i] and len(record_lidar) == 1:
                    pixels = record_lidar[0].project_points_to_image(camera_intrinsics=intrinsics_dir[canvas_order[i]],
                                                                     extrinsic=extrinsic_dir[canvas_order[i]],
                                                                     distortion_vector=distortion_vector_dir[canvas_order[i]])
                    pixels_resized = np.zeros((pixels.shape[0], 2))
                    pixels_resized[:,0] = pixels[:,0] * (cell_width / img.shape[1])
                    pixels_resized[:,1] = pixels[:,1] * (cell_height / img.shape[0])

                    if intensity:
                        colors = record_lidar[0].get_color_for_intensity(max=2000)
                    else:
                        colors = record_lidar[0].get_color_for_distance(color_palette='jet', loop = loop)
                    for j in range(len(pixels)):
                        resized = cv2.circle(resized,
                                             (int(pixels_resized[j, 0]), int(pixels_resized[j, 1])), radius = 0, color=(colors[j,2], colors[j,1], colors[j,0]), thickness=-1)
    
                if with_anns[i]:
                    for box in boxes_dir[canvas_order[i]]:
                        if box.instance_token in list_instances or list_instances == []:
                            if filter_category == [] or box.name in filter_category:
                                resized = cv2.rectangle(resized,
                                                        (int(box.bbox[0] * (cell_width / img.shape[1])), int(box.bbox[1] * (cell_height / img.shape[0]))),
                                                        (int(box.bbox[2] * (cell_width / img.shape[1])), int(box.bbox[3] * (cell_height / img.shape[0]))),
                                                        box.color, 2)
                                if with_instance:
                                    resized = cv2.putText(resized,
                                                        box.instance_token[:5]+'..',
                                                        (int(box.bbox[0] * (cell_width / img.shape[1])), int(box.bbox[1] * (cell_height / img.shape[0]))),
                                                        0, 1, box.color, 2)

                canvas[y_start:y_end, x_start:x_end] = resized

            if type(canvas_order) != list:
                if with_lidar[i] and with_anns[i]:
                    img = list(imgs.values())[count_imgs].render(show = False, lidar = record_lidar[0], boxes = boxes, camera_intrinsics = list(intrinsics_dir.values())[count_imgs], extrinsic = list(extrinsic_dir.values())[count_imgs])
                elif not with_lidar[i] and with_anns[i]:
                    img = list(imgs.values())[count_imgs].render(show = False, boxes = boxes, camera_intrinsics = list(intrinsics_dir.values())[count_imgs], extrinsic = list(extrinsic_dir.values())[count_imgs])
                elif with_lidar[i] and not with_anns[i]:
                    img = list(imgs.values())[count_imgs].render(show = False, lidar = record_lidar[0], camera_intrinsics = list(intrinsics_dir.values())[count_imgs], extrinsic = list(extrinsic_dir.values())[count_imgs])
                else:
                    img = list(imgs.values())[count_imgs].render(show = False)

                row = i // num_cols
                col = i % num_cols
                x_start = col * cell_width
                x_end = x_start + cell_width
                y_start = row * cell_height
                y_end = y_start + cell_height

                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                canvas[y_start:y_end, x_start:x_end] = cv2.resize(img, (cell_width, cell_height))
                count_imgs += 1
        
        geometries = None

        if len(record_lidar) == 1:
            spheres = []
            geometries = [record_lidar[0].point_cloud]
            geo_spheres = [record_lidar[0].point_cloud]
            color_radar= [(1,0,0), (0,0,1)]
            for i in range(len(record_radar)):
                radar_i = record_radar[i]
                radar_i.transform(radar_extrinsics[i])
                spheres.extend(radar_i.get_sphere_geometry(min = record_lidar[0].min(),
                                                     max = record_lidar[0].max(),
                                                     radius = 0.25,
                                                     color_palette=color_radar[i]))
                geometries.append(radar_i)
            geo_spheres.extend(spheres)
            if with_anns[0]:
                geo_spheres.extend([i.get_OBB() for i in boxes_dir['LIDAR']])
        if show:
            cv2.namedWindow(f'{window_name}', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'{window_name}', width, height)
            cv2.imshow(f'{window_name}', canvas)
            cv2.waitKey(0)

            cv2.destroyWindow(f'{window_name}')
            if show_pcd and len(record_lidar) == 1:
                record_lidar[0].render(other_geometries = geo_spheres[1:], loop = loop)
        return canvas, geometries
    
    def render_map(self,
                   scene_token: str,
                   gps: bool = True,
                   ins: bool = True,
                   inline: bool =  False,
                   browser: bool =  True,
                   save_path: str = None):
        
        scene_record = self.useful.get('scene', scene_token)
        sample_record = self.useful.get('sample', scene_record['first_sample_token'])
        ego_pose_record = self.useful.get('ego_pose', sample_record['ego_pose_token'])

        map = folium.Map(location=[ego_pose_record['GPS']['lat_deg'], ego_pose_record['GPS']['lon_deg']], zoom_start=20)
        folium.Marker([ego_pose_record['GPS']['lat_deg'], ego_pose_record['GPS']['lon_deg']], popup='Start').add_to(map)
        if ins:
            sample_record = self.useful.get('sample', scene_record['first_sample_token'])
            ego_pose_record = self.useful.get('ego_pose', sample_record['ego_pose_token'])
            locations_ins = []
            for i in range(scene_record['nbr_samples']):
                ego_pose_record = self.useful.get('ego_pose', sample_record['ego_pose_token'])
                locations_ins.append((float(ego_pose_record['INS']['lat_deg']), float(ego_pose_record['INS']['lon_deg'])))
                if sample_record['next'] == '':
                    break
                sample_record = self.useful.get('sample', sample_record['next'])
            folium.PolyLine(locations_ins, color='red').add_to(map)
        if gps:
            sample_record = self.useful.get('sample', scene_record['first_sample_token'])
            ego_pose_record = self.useful.get('ego_pose', sample_record['ego_pose_token'])
            locations_gps = []
            for i in range(scene_record['nbr_samples']):
                ego_pose_record = self.useful.get('ego_pose', sample_record['ego_pose_token'])
                locations_gps.append((float(ego_pose_record['GPS']['lat_deg']), float(ego_pose_record['GPS']['lon_deg'])))
                if sample_record['next'] == '':
                    break
                sample_record = self.useful.get('sample', sample_record['next'])
            folium.PolyLine(locations_gps, color='blue').add_to(map)

        if inline:
            display(map)
        elif browser:
            map.save(f'{scene_token}_map.html')
            webbrowser.open(f'{scene_token}_map.html')
            if os.path.exists(f'{scene_token}_map.html'):
                os.remove(f'{scene_token}_map.html')
        if save_path is not None:
            map.save(save_path)
                

    def get_matrices(self,
                     calibrated_sensor_token: str) -> dict:
        
        '''
        Returns the camera intrinsic matrix and the extrinsic matrix of a calibrated_sensor.
        '''
        cs_record = self.useful.get('calibrated_sensor', calibrated_sensor_token)
        if cs_record['camera_intrinsic'] ==[]:
            camera_intrinsic = []
        else:
            camera_intrinsic = np.array(cs_record['camera_intrinsic']).reshape(3,3)
        translation = np.array(cs_record['translation'])
        rotation = Quaternion(np.array(cs_record['rotation'])).rotation_matrix
        extrinsic = np.eye(4)
        extrinsic[:3,:3] = rotation
        extrinsic[:3,3] = translation
        distortion_vector = np.array(cs_record['distortion_vector'])

        return camera_intrinsic, extrinsic, distortion_vector
    
    def get_categories(self):
        '''
        Returns a dictionary with the categories of the dataset.
        ''' 
        categories = self.useful.category
        category_dict = {}
        for i in range(len(categories)):
            category_dict[categories[i]['name']] = i
        return category_dict
    
    def render_sample_bev(self,
                          sample_token: str,
                          width: int = 1800,
                          height: int = 800,
                          with_anns: bool = True,
                          bev_max_range: float = 150.0,
                          bev_resolution: float = 0.1,
                          intensity: bool = False,
                          show: bool = True) -> np.ndarray:
        """
        Renders the BEV + 6-camera composite for a single sample and returns it as a BGR ndarray.
        :param sample_token: Token of the sample to render.
        :param width: Total composite width in pixels (BEV takes left 1/3, cameras right 2/3).
        :param height: Total composite height in pixels.
        :param with_anns: Draw 2D/BEV annotations.
        :param bev_max_range: Forward range in metres for the BEV.
        :param bev_resolution: Metres per pixel for the BEV.
        :param intensity: Colour LiDAR points by intensity instead of distance.
        :param show: Display the result in a cv2 window.
        :return: BGR ndarray of shape (height, width, 3).
        """
        sample_record = self.useful.get('sample', sample_token)
        left_w = width // 3
        right_w = width - left_w
        cell_w = right_w // 3
        cell_h = height // 2
        camera_order = ['WIDE_LEFT', 'NARROW', 'WIDE_RIGHT', 'LWIR', 'POLARIMETRIC', 'SWIR']

        lidar_sd_token = sample_record['data'].get('LIDAR')
        if lidar_sd_token is not None:
            lidar_path = self.useful.get_sample_data_path(lidar_sd_token)
            lidar_pc = LidarPointCloud(data_path=lidar_path, format='USEFUL')
            boxes = self.useful.get_boxes(sample_token)
            bev_img = self._render_bev_image(lidar_pc, boxes,
                                             max_range=bev_max_range,
                                             resolution=bev_resolution,
                                             with_anns=with_anns,
                                             intensity=intensity)
        else:
            bev_img = np.zeros((height, left_w, 3), dtype=np.uint8)
        bev_resized = cv2.resize(bev_img, (left_w, height))

        cells = []
        for ch in camera_order:
            sd_token = sample_record['data'].get(ch)
            if sd_token is None:
                cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
                continue
            data_path = self.useful.get_sample_data_path(sd_token)
            sd_record = self.useful.get('sample_data', sd_token)
            boxes_2d = self.useful.get_boxes_2d(sd_token) if with_anns else []
            modality = sd_record['sensor_modality']
            if modality == 'rgb':
                img = RGBImage(data_path=data_path).render(
                    show=False, boxes=boxes_2d, bbox_2d=True)
            elif modality == 'thermal':
                img = ThermalImage(data_path).render(
                    show=False, boxes=boxes_2d, bbox_2d=True)
            elif modality == 'swir':
                img = SWIRImage(data_path).render(
                    show=False, boxes=boxes_2d, bbox_2d=True)
            elif modality == 'polarimetric':
                pol = PolarimetricImage(data_path).getMode('RGB')[:, :, [2, 1, 0]]
                pol = (pol / (np.max(pol) + 1e-6) * 255).astype(np.uint8)
                img = RGBImage(img=pol).render(show=False, boxes=boxes_2d, bbox_2d=True)
            else:
                img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cells.append(cv2.resize(img, (cell_w, cell_h)))

        row1 = np.hstack(cells[:3])
        row2 = np.hstack(cells[3:])
        frame = np.hstack([bev_resized, np.vstack([row1, row2])])

        if show:
            cv2.imshow('Sample BEV', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return frame

    def render_lidar_bev(self,
                         sample_token: str,
                         bev_max_range: float = 150.0,
                         bev_resolution: float = 0.1,
                         with_anns: bool = True,
                         intensity: bool = False,
                         show: bool = True) -> np.ndarray:
        """
        Renders the BEV of the LiDAR point cloud for a single sample.
        :param sample_token: Token of the sample to render.
        :param bev_max_range: Forward range in metres.
        :param bev_resolution: Metres per pixel.
        :param with_anns: Draw 3D box footprints.
        :param intensity: Colour points by intensity instead of distance.
        :param show: Display the result in a cv2 window.
        :return: BGR ndarray of the BEV image.
        """
        sample_record = self.useful.get('sample', sample_token)
        lidar_sd_token = sample_record['data'].get('LIDAR')
        if lidar_sd_token is None:
            return np.zeros((int(bev_max_range / bev_resolution), 10, 3), dtype=np.uint8)
        lidar_path = self.useful.get_sample_data_path(lidar_sd_token)
        lidar_pc = LidarPointCloud(data_path=lidar_path, format='USEFUL')
        boxes = self.useful.get_boxes(sample_token) if with_anns else []
        bev = self._render_bev_image(lidar_pc, boxes,
                                     max_range=bev_max_range,
                                     resolution=bev_resolution,
                                     with_anns=with_anns,
                                     intensity=intensity)
        if show:
            cv2.imshow('LiDAR BEV', bev)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return bev

    def _render_bev_image(self,
                          lidar_pc,
                          boxes,
                          max_range: float = 150.0,
                          resolution: float = 0.1,
                          with_anns: bool = True,
                          intensity: bool = False) -> np.ndarray:
        """
        Renders a Bird's Eye View image of a LiDAR point cloud with optional 3D box footprints.
        The view covers the forward hemisphere (60 deg HFOV): x in [0, max_range],
        y in [-max_range*tan(30deg), max_range*tan(30deg)].
        Forward direction is up in the resulting image.
        :return: BGR ndarray (img_h, img_w, 3).
        """
        y_half = max_range * np.tan(np.radians(30))
        img_h = int(max_range / resolution)
        img_w = int(2 * y_half / resolution)
        bev = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        points = lidar_pc.points  # (N, 3+)
        mask = (points[:, 0] > 0) & (points[:, 0] < max_range) & (np.abs(points[:, 1]) < y_half)
        pts = points[mask]

        if len(pts) > 0:
            if intensity:
                colors = lidar_pc.get_color_for_intensity(max=2000)[mask]
            else:
                colors = lidar_pc.get_color_for_distance(color_palette='jet', loop=10)[mask]
            py = np.clip(((max_range - pts[:, 0]) / resolution).astype(int), 0, img_h - 1)
            px = np.clip(((y_half - pts[:, 1]) / resolution).astype(int), 0, img_w - 1)
            bev[py, px] = colors[:, :3]

        if with_anns:
            for box in boxes:
                if not (0 < box.center[0] < max_range and abs(box.center[1]) < y_half):
                    continue
                # Bottom footprint corners (indices 4-7): (4, 3) → take X, Y columns
                corners = box.corners()[4:, :2]
                pts_px = np.array([
                    [int((y_half - c[1]) / resolution), int((max_range - c[0]) / resolution)]
                    for c in corners
                ], dtype=np.int32)
                cv2.polylines(bev, [pts_px.reshape(-1, 1, 2)], isClosed=True,
                              color=box.color, thickness=2)
                # Forward direction: center → midpoint of front edge (bottom corners 4, 5)
                cx_px = int((y_half - box.center[1]) / resolution)
                cy_px = int((max_range - box.center[0]) / resolution)
                front_mid = ((pts_px[0] + pts_px[1]) // 2)
                cv2.line(bev, (cx_px, cy_px), tuple(front_mid), box.color, 2)

        # --- Metric axes overlay ---
        grid_color = (50, 50, 50)
        label_color = (180, 180, 180)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        # Forward (depth) grid lines every 25 m
        for x_m in range(25, int(max_range) + 1, 25):
            py_line = int((max_range - x_m) / resolution)
            if 0 <= py_line < img_h:
                cv2.line(bev, (0, py_line), (img_w - 1, py_line), grid_color, 1)
                cv2.putText(bev, f'{x_m}m', (5, py_line - 3),
                            font, font_scale, label_color, 1, cv2.LINE_AA)
        # Lateral (y) grid lines every 25 m
        y_step = 25
        y_ticks = range(-int(y_half // y_step) * y_step,
                        int(y_half // y_step) * y_step + 1, y_step)
        for y_m in y_ticks:
            px_line = int((y_half - y_m) / resolution)
            if 0 <= px_line < img_w:
                cv2.line(bev, (px_line, 0), (px_line, img_h - 1), grid_color, 1)
                label = f'{y_m}m'
                (tw, _th), _ = cv2.getTextSize(label, font, font_scale, 1)
                cv2.putText(bev, label, (px_line - tw // 2, img_h - 5),
                            font, font_scale, label_color, 1, cv2.LINE_AA)
        # Ego marker at bottom-centre
        ego_px = int(y_half / resolution)
        cv2.drawMarker(bev, (ego_px, img_h - 1), (255, 255, 255),
                       cv2.MARKER_CROSS, 12, 2)

        return bev

    def render_scene(self,
                     scene_token: str,
                     save_path: str,
                     fps: int = 7,
                     width: int = 1800,
                     height: int = 800,
                     with_anns: bool = True,
                     bev_max_range: float = 150.0,
                     bev_resolution: float = 0.1,
                     intensity: bool = False):
        """
        Renders a scene to a video. Each frame has:
          - Left third: Bird's Eye View of LiDAR with projected 3D box footprints.
          - Right two thirds: 2x3 grid of cameras [WIDE_LEFT, NARROW, WIDE_RIGHT;
                                                    LWIR, POLARIMETRIC(RGB), SWIR].
        :param scene_token: Scene token.
        :param save_path: Output video path (e.g. 'scene.mp4').
        :param fps: Frames per second.
        :param width: Total output video width in pixels.
        :param height: Total output video height in pixels.
        :param with_anns: Whether to draw 2D/BEV annotations.
        :param bev_max_range: Forward range in metres for the BEV (default 150).
        :param bev_resolution: Metres per pixel for the BEV (default 0.1).
        :param intensity: Colour LiDAR points by intensity instead of distance.
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        scene_record = self.useful.get('scene', scene_token)
        sample_token = scene_record['first_sample_token']

        left_w = width // 3
        right_w = width - left_w
        cell_w = right_w // 3
        cell_h = height // 2
        camera_order = ['WIDE_LEFT', 'NARROW', 'WIDE_RIGHT', 'LWIR', 'POLARIMETRIC', 'SWIR']

        for _ in tqdm.tqdm(range(scene_record['nbr_samples']), 'Writing video...'):
            sample_record = self.useful.get('sample', sample_token)

            # --- BEV (left panel) ---
            lidar_sd_token = sample_record['data'].get('LIDAR')
            if lidar_sd_token is not None:
                lidar_path = self.useful.get_sample_data_path(lidar_sd_token)
                lidar_pc = LidarPointCloud(data_path=lidar_path, format='USEFUL')
                boxes = self.useful.get_boxes(sample_record['token'])
                bev_img = self._render_bev_image(lidar_pc, boxes,
                                                 max_range=bev_max_range,
                                                 resolution=bev_resolution,
                                                 with_anns=with_anns,
                                                 intensity=intensity)
            else:
                bev_img = np.zeros((height, left_w, 3), dtype=np.uint8)
            bev_resized = cv2.resize(bev_img, (left_w, height))

            # --- Camera grid (right panel) ---
            cells = []
            for ch in camera_order:
                sd_token = sample_record['data'].get(ch)
                if sd_token is None:
                    cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
                    continue
                data_path = self.useful.get_sample_data_path(sd_token)
                sd_record = self.useful.get('sample_data', sd_token)
                boxes_2d = self.useful.get_boxes_2d(sd_token) if with_anns else []
                modality = sd_record['sensor_modality']
                if modality == 'rgb':
                    img = RGBImage(data_path=data_path).render(
                        show=False, boxes=boxes_2d, bbox_2d=True)
                elif modality == 'thermal':
                    img = ThermalImage(data_path).render(
                        show=False, boxes=boxes_2d, bbox_2d=True)
                elif modality == 'swir':
                    img = SWIRImage(data_path).render(
                        show=False, boxes=boxes_2d, bbox_2d=True)
                elif modality == 'polarimetric':
                    pol = PolarimetricImage(data_path).getMode('RGB')[:, :, [2, 1, 0]]
                    pol = (pol / (np.max(pol) + 1e-6) * 255).astype(np.uint8)
                    img = RGBImage(img=pol).render(show=False, boxes=boxes_2d, bbox_2d=True)
                else:
                    img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cells.append(cv2.resize(img, (cell_w, cell_h)))

            row1 = np.hstack(cells[:3])
            row2 = np.hstack(cells[3:])
            right_grid = np.vstack([row1, row2])

            frame = np.hstack([bev_resized, right_grid])
            out.write(frame)

            sample_token = sample_record['next']
            if sample_token == '':
                break

        out.release()
        print(f'Video saved at {save_path}')
        
