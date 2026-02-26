import numpy as np
import copy
import open3d as o3d
import cv2
import os
import pathlib
import matplotlib.pyplot as plt
import struct

from pyquaternion import Quaternion
from matplotlib.axes import Axes
#from .geometry_utils import view_points, box_in_image

class PointCloud:
    def __len__(self) -> int:
        return len(self.points)
    
    def __repr__(self) -> str:
        repr_str = f'Data path: {self.data_path}\nNumber of points: {len(self)}\nMax: {self.points[self.max(return_idx=True)[1],:]} ({self.max():.3f})\nMin: {self.points[self.min(return_idx=True)[1],:]} ({self.min():.3f})\n Points: {self.points}'
        return repr_str
    
    def max(self,
            mode: str = 'Euclidean',
            return_idx: bool = False) -> tuple:
        '''
        Returns the distance value of the furthest point to the origin.
        '''
        available_modes = ['Euclidean', 'X', 'Y', 'Z']
        assert mode in available_modes, f'{mode} not available. Try {available_modes}'
        if mode == 'Euclidean':
            distances = np.sqrt(self.points[:,0]**2 + self.points[:,1]**2 + self.points[:,2]**2)
        elif mode == 'X':
            distances = np.abs(self.points[:,0])
        elif mode == 'Y':
            distances = np.abs(self.points[:,1])
        elif mode == 'Z':
            distances = np.abs(self.points[:,2])
        
        if return_idx:
            return max(distances), np.argmax(distances)
        else:
            return max(distances)
    
    def min(self,
            mode: str = 'Euclidean',
            return_idx: bool = False) -> tuple:
        '''
        Returns the Euclidean Distance value of the closest point to the origin.
        '''
        available_modes = ['Euclidean', 'X', 'Y', 'Z']
        assert mode in available_modes, f'{mode} not available. Try {available_modes}'
        if mode == 'Euclidean':
            distances = np.sqrt(self.points[:,0]**2 + self.points[:,1]**2 + self.points[:,2]**2)
        elif mode == 'X':
            distances = np.abs(self.points[:,0])
        elif mode == 'Y':
            distances = np.abs(self.points[:,1])
        elif mode == 'Z':
            distances = np.abs(self.points[:,2])
        
        if return_idx:
            return min(distances), np.argmin(distances)
        else:
            return min(distances)
   
    def distances(self) -> np.ndarray:

        distances = np.sqrt(self.points[:,0]**2 + self.points[:,1]**2 + self.points[:,2]**2)
        return distances

    def filter_distance(self,
                        min: float = 0,
                        max: float = np.inf,
                        mode: str = 'Euclidean') -> None:
        
        assert max > min, f'Max ({max}) distance should be larger than Min ({min}) distance.'

        if mode == 'Euclidean':
            distances = np.sqrt(self.points[:,0]**2 + self.points[:,1]**2 + self.points[:,2]**2)
        self.points = self.points[np.logical_and(distances > min , distances < max), :]
        self.intensity = self.intensity[np.logical_and(distances > min , distances < max)]
        self.point_cloud.points = o3d.utility.Vector3dVector(self.points)
        if np.asarray(self.point_cloud.colors).size != 0: 
            colors = np.asarray(self.point_cloud.colors)
            colors = colors[np.logical_and(distances > min, distances < max), :]
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    def get_color_for_distance(self,
                               color_palette: str = 'jet',
                               min: float = None,
                               max: float = None,
                               loop: float = None):
        distances = self.distances()
        if loop is not None:
            distances = distances % loop
            normalized_distance = distances / loop

        if min == None:
            min = self.min() 
        if max == None:
            max = self.max()
        if loop is None:
            normalized_distance = (distances - min) / (max - min)

        # Apply colormap to normalized distance
        jet_cmap = plt.get_cmap(color_palette)
        colors = jet_cmap(normalized_distance)
        colors = colors[:,:3] * 255
        return colors
    
    def get_color_for_intensity(self,
                               color_palette: str = 'jet',
                               min: float = None,
                               max: float = None):
        if min == None:
            min = np.min(self.intensity)
        if max == None:
            max = np.max(self.intensity)
        normalized_distance = (self.intensity - min) / (max - min)

        # Apply colormap to normalized distance
        jet_cmap = plt.get_cmap(color_palette)
        colors = jet_cmap(normalized_distance)
        colors = colors[:,:3] * 255
        return colors
    
    def get_sphere_geometry(self,
                            radius: float = 0.5,
                            min: float = None,
                            max : float = None,
                            color_palette: str = 'jet',
                            show: bool = False,
                            vis: o3d.visualization.Visualizer = None) -> list:
        sphere_points = []
        if type(color_palette) == str:
            colors = np.array(self.get_color_for_distance(color_palette=color_palette, min = min, max = max)) / 255
        for i in range(len(self)):
            mesh = o3d.geometry.TriangleMesh.create_tetrahedron(radius = radius)
            mesh.translate(self.points[i, :])
            if type(color_palette) == str:
                mesh.paint_uniform_color(colors[i,:])
            else:
                mesh.paint_uniform_color(color_palette)
            sphere_points.append(mesh)
        if show:

            if vis is None:
                vis = o3d.visualization.Visualizer()
                vis.create_window()
            for i in range(len(sphere_points)):
                vis.add_geometry(sphere_points[i])
            vis.run()
            vis.destroy_window()

        return sphere_points
    
    def get_axis(self,
                 length: float = 1.0) -> o3d.geometry.LineSet:
        """
        Returns a LineSet representing the 3D axes.
        """
        # Define points: origin + one point in each direction
        points = np.array([
            [0, 0, 0],        # Origin
            [length, 0, 0],   # X-axis
            [0, length, 0],   # Y-axis
            [0, 0, length]    # Z-axis
        ])

        # Define lines: from origin to X, Y, Z
        lines = np.array([
            [0, 1],  # X-axis
            [0, 2],  # Y-axis
            [0, 3]   # Z-axis
        ])

        # Colors for each axis line: RGB
        colors = np.array([
            [1, 0, 0],  # Red for X-axis
            [0, 1, 0],  # Green for Y-axis
            [0, 0, 1]   # Blue for Z-axis
        ])

        # Create LineSet geometry
        axis_lines = o3d.geometry.LineSet()
        axis_lines.points = o3d.utility.Vector3dVector(points)
        axis_lines.lines = o3d.utility.Vector2iVector(lines)
        axis_lines.colors = o3d.utility.Vector3dVector(colors * 255)
        return axis_lines
    
    def translate(self,
                  translation: np.ndarray) -> None:
        
        self.points = self.points + translation
        self.point_cloud.points = o3d.utility.Vector3dVector(self.points)
    
    def rotate(self,
               rotation: np.ndarray) -> None:
        
        assert rotation.shape == (3,3), 'Rotation has to be a 3 by 3 matrix'
        self.points = (rotation @ self.points.T).T
        self.point_cloud.points = o3d.utility.Vector3dVector(self.points)

    def transform(self,
                  transformation: np.ndarray) -> None:
        assert transformation.shape == (4,4), 'Transformation has to be a 4 by 4 matrix'
        
        homogeneous_points = np.hstack((self.points, np.ones((self.points.shape[0], 1))))
        transformed_points = (transformation @ homogeneous_points.T).T

        self.points = transformed_points[:,:3]
        self.point_cloud.points = o3d.utility.Vector3dVector(self.points)
        
    def transform_quat(self,
                       quat: np.ndarray,
                       translation: np.ndarray) -> None:
        H = np.eye(4)
        H[:3,:3] = Quaternion(quat).rotation_matrix
        H[:3,3] = np.array(translation)

        homogeneous_points = np.hstack((self.points, np.ones((self.points.shape[0], 1))))
        transformed_points = (H @ homogeneous_points.T).T

        self.points = transformed_points[:,:3]
        self.point_cloud.points = o3d.utility.Vector3dVector(self.points)
    
    def render(self,
               vis: o3d.visualization.Visualizer = None,
               other_geometries: list = [],
               img: np.ndarray = None,
               extrinsic: np.ndarray = None,
               camera_intrinsics: np.ndarray = None,
               distortion_vector: np.ndarray = None,
               intensity: bool = False,
               loop: float = None,
               axis: float = 1.0) -> None:
        """
        Renders the point cloud with optional image fusion, intensity mapping, or distance-based coloring.

        Args:
            vis: Optional Open3D visualizer instance. If None, a new window is created.
            other_geometries: List of additional Open3D geometries to display.
            img: Input RGB image for point cloud color projection.
            extrinsic: 4x4 transformation matrix from world/lidar coordinates to camera coordinates.
            camera_intrinsics: 3x3 camera intrinsic matrix. Required if 'img' is provided.
            distortion_vector: Lens distortion coefficients for precise point projection.
            intensity: If True, maps point cloud colors based on intensity values.
            loop: If provided, scales distance-based coloring using this value as a cycle period.
            axis: Length of the coordinate axis to render. Set to 0 to disable.

        Returns:
            None: Opens a blocking Open3D visualization window.
        """
        
        if img is not None:
            if camera_intrinsics is None:
                assert camera_intrinsics is not None, "Camera intrinsic parameters are required to fuse point cloud with image."
            if extrinsic is None:
                print('No extrinsic matrix has been defined, used identity')
                extrinsic = np.eye(4)
            
            pixels = self.project_points_to_image(camera_intrinsics= camera_intrinsics,
                                                  extrinsic= extrinsic,
                                                  distortion_vector= distortion_vector)
            
            if intensity:
                colors_int = self.get_color_for_intensity(max = 2000)
            else:
                colors_dist = self.get_color_for_distance(loop=loop)
            h, w = img.shape[0], img.shape[1]
            colors = np.zeros_like(pixels)
            for i in range(len(pixels)):
                if pixels[i,0] > 0 and pixels[i,0] < w and pixels[i,1] > 0 and pixels[i,1] < h and pixels[i,2] > 0:
                    colors[i,:] = img[int(pixels[i,1]), int(pixels[i,0])] / 255
                else:
                    if intensity:
                        colors[i,:] = colors_int[i,:] / 255
                    else:
                        colors[i,:] = colors_dist[i,:] / 255
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        elif intensity:
            colors = self.get_color_for_intensity(max = 2000) / 255
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        elif loop is not None:
            colors = self.get_color_for_distance(loop = loop) / 255
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        if vis is None:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
        
        vis.add_geometry(self.point_cloud)
        if axis > 0:
            axis_geometry = self.get_axis(length=axis)
            vis.add_geometry(axis_geometry)
            
        if len(other_geometries) != 0:
            for geo in other_geometries:
                vis.add_geometry(geo)
                vis.update_renderer()
        
        vis.run()
        vis.destroy_window()

    def project_points_to_image(self,
                                camera_intrinsics: np.ndarray,
                                extrinsic: np.ndarray,
                                distortion_vector: np.ndarray = None,
                                ROI: list = None) -> np.ndarray:
        """
        Projects 3D point cloud points onto a 2D image plane using camera parameters.

        The projection follows the standard pinhole camera model: 
        1. Transform points to camera coordinate system using extrinsic matrix.
        2. Project to image plane using intrinsic matrix.
        3. Apply radial/tangential distortion if a distortion vector is provided.
        4. Optionally filter points within a Region of Interest (ROI).

        Args:
            camera_intrinsics: 3x3 matrix representing the camera calibration parameters.
            extrinsic: 4x4 rigid transformation matrix (R|t) from Lidar to Camera frame.
            distortion_vector: Optional coefficients for camera lens distortion.
            ROI: Optional list [y_min, x_min, y_max, x_max] to filter points within a 2D bounding box.

        Returns:
            If ROI is None:
                pixels: (N, 3) array where [u, v] are image coordinates and [z] is the depth.
            If ROI is provided:
                tuple: (pixels, filtered_points, mask_indices)
                    - pixels: (M, 3) array of points within ROI.
                    - filtered_points: (M, 3) original 3D coordinates of points within ROI.
                    - mask_indices: Boolean mask used for filtering.

        Raises:
            AssertionError: If intrinsic or extrinsic matrices do not have the required dimensions.
        """
        assert camera_intrinsics.shape == (3,3), f'Shape {camera_intrinsics.shape}, does not match expected shape (3,3) for camera_intrinsics'
        assert extrinsic.shape == (4,4), f'Shape {extrinsic.shape}, does not match expected shape (4,4) for extrinsic'

        homogeneous_points = np.hstack((self.points, np.ones((self.points.shape[0], 1))))
        transformed_points = (extrinsic @ homogeneous_points.T).T

        points = transformed_points[:,:3]

        projected_points = (camera_intrinsics @ points.T).T

        pixels = np.empty_like(projected_points)
        pixels[:,0] = projected_points[:,0] / projected_points[:,2]
        pixels[:,1] = projected_points[:,1] / projected_points[:,2]
        pixels[:,2] = projected_points[:,2]

        if distortion_vector is not None:
            #assert len(distortion_vector) == 5, f'The number of components in distortion_vector ({len(distortion_vector)}) is not what is expected (5)'
            pixels = self.distortImagePoint(pixels, camera_intrinsics, distortion_vector)

        if ROI is not None:
            idx = (pixels[:, 0] >= ROI[1]) & (pixels[:, 0] <= ROI[3]) & (pixels[:, 1] >= ROI[0]) & (pixels[:, 1] <= ROI[2])
            points = self.points[idx,:]
            pixels = pixels[idx,:]
            return pixels, points, idx
        else:
            return pixels

    def beamagine_color_palette(self,
                                max_color_value: int = 5000,
                                min_color_value: int = 0):

        R = np.array([0,100, 200, 220, 242, 231, 226])
        G = np.array([176, 189, 211, 160, 134, 70, 3])
        B = np.array([176, 85, 1, 0, 0, 20, 47, 226])

        lut_scale = 255.0 / (max_color_value - min_color_value)
        color_value = np.round(self.distances() - min_color_value) * lut_scale
        color_value  = color_value % 255

        MAX_COLOR_LEVEL = 7

        op = np.uint8(color_value / (255/MAX_COLOR_LEVEL))

        Lo = np.array([MAX_COLOR_LEVEL - 1 if i == 0 else i - 1 for i in op ])
        Lf = op
        red = (R[Lo] + 0.0277*(R[Lf] - R[Lo]) * color_value).reshape(len(op), 1)
        green = (G[Lo] + 0.0277*(G[Lf] - G[Lo]) * color_value).reshape(len(op), 1)
        blue = (B[Lo] + 0.0277*(B[Lf] - B[Lo]) * color_value).reshape(len(op), 1)

        return np.hstack((red, green, blue))
    

    def project_to_plane(self,
                         planeParams: np.ndarray,
                         direction: np.ndarray) -> np.ndarray:
        '''
        Projects the 3D points in a point cloud onto a plane defined by a set of plane parameters,
        following a specific direction.

        :param @points: (np.ndarray) N by 3 matrix with point coordinates
        :param @planeParams: (np.ndarray) a 4x1 vector containing the plane parameters [a, b, c, d],
                            where ax + by + cz + d = 0 is the equation of the plane.
        :param @direction: (np.ndarray) a 3x1 vector representing the direction onto which points will be projected.
        :returns @projectedPoints: (np.ndarray) a N by 3 matrix containing the coordinates of the projected points onto the plane.
        '''
        # Normalize the direction vector
        direction /= np.linalg.norm(direction)

        # Calculate the orthogonal projection matrix onto the plane
        plane_normal = planeParams[:3] / np.linalg.norm(planeParams[:3])
        projection_matrix = np.eye(3) - np.outer(plane_normal, plane_normal)

        # Project the direction vector onto the plane
        projected_direction = projection_matrix.dot(direction)

        # Project the points onto the plane along the projected direction
        distances = np.dot(self.points, projected_direction)
        projected_points = self.points - np.outer(distances, projected_direction)

        return projected_points

    def select_points(self) -> tuple:
        """
        Opens an interactive Open3D window to manually select points from the point cloud.

        The selection process is interactive:
        - Press 'Shift + Left Click' to select a point.
        - The indices and 3D coordinates of the selected points are captured upon closing the window.

        Returns:
            tuple: A tuple containing:
                - picked_points_coordinates (np.ndarray): 3D coordinates (x, y, z) of the selected points.
                - picked_points (np.ndarray): Indices of the selected points in the original point cloud.
        """
        # Create a visualizer object
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()

        # Add the point cloud to the visualizer
        vis.add_geometry(self.point_cloud)

        # Run the visualizer (this allows you to interactively select points)
        vis.run()  # shift + left click

        # Destroy the visualizer window
        vis.destroy_window()

        # Get the picked points indices
        picked_points = np.array(vis.get_picked_points())
        print("Picked points indices:", picked_points)

        # Get the coordinates of the picked points
        picked_points_coordinates = np.array([self.point_cloud.points[idx] for idx in picked_points])

        return picked_points_coordinates, picked_points
    
    def ransac(self,
               distance_threshold: float = 200,
               ransac_n: int = 3,
               num_iterations: int = 1000,
               visualization: bool = False):
        '''
        Uses Open3d RANSAC algorithm to obtain parameters or to visualize the new PointCloud.
        - Inputs:
            * distance_threshold (int|float = 200) [Optional]: distance from plane threshold to consider a point an inlier.
            * ransac_n (int = 3) [Optional]:  Number of initial points to be considered inliers in each iteration
            * num_iterations (int = 1000) [Optional]:  Number of iterations
        - Outputs:
            * params (np.ndarray):  4 by 1 matrix with plane parameters
            * inlier_points (np.ndarray): N by 3 matrix with inlier point coordinates
            * outlier_points (np.ndarray): N by 3 matrix with outlier point coordinates
        '''
        params, inliers = self.point_cloud.segment_plane(distance_threshold = distance_threshold,
                                                         ransac_n = ransac_n,
                                                         num_iterations = num_iterations)
        
        inlier_cloud = self.point_cloud.select_by_index(inliers)
        outlier_cloud = self.point_cloud.select_by_index(inliers, invert = True)

        inlier_points = np.asarray(inlier_cloud.points)
        outlier_points = np.asarray(outlier_cloud.points)

        if visualization:
            def draw_direction(start_point: np.ndarray,
                   direction:np.ndarray,
                   elongation: float,
                   color: np.ndarray = np.array([0,1,0])) -> o3d.geometry.LineSet:
    
                end_point = start_point + direction * elongation
                line = o3d.geometry.LineSet(points= o3d.utility.Vector3dVector(np.array([start_point,end_point])),lines = o3d.utility.Vector2iVector([[0,1]]))
                line.paint_uniform_color(color)
                return line
            inlier_cloud.paint_uniform_color([0,0,0])
            geometries = [inlier_cloud]
            geometries.append(outlier_cloud)
            geometries.append(draw_direction(np.mean(inlier_points, axis = 0), params[:-1], 500))
            o3d.visualization.draw_geometries(geometries)
        return params, inlier_points, outlier_points
    
    def get_points_from_box(self,
                            center: np.ndarray,
                            size: np.ndarray,
                            yaw: np.ndarray) -> np.ndarray:
        '''
        center : (3,) array-like
            Center of the box [cx, cy, cz].
        size : (3,) array-like
            Size of the box [dx, dy, dz] (length along x, y, z in box frame).
        yaw : float
            Rotation around Z axis in radians (right-hand rule).

        Returns
        -------
        inside_points : (M, 3) np.ndarray
            Points that lie inside the box.
        mask : (N,) np.ndarray of bool
            Boolean mask indicating which points are inside.

        '''
        pc_copy = copy.deepcopy(self)
        points = pc_copy.points
        intensity = pc_copy.intensity
        # Translate points so box center is at origin
        pts_rel = points - center
        # Rotation matrix to align points to box frame (inverse of box yaw)
        cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])
        # Rotate points into box frame
        pts_rot = pts_rel @ R.T
        
        # Check box limits (axis-aligned in box frame)
        half_size = size / 2.0
        mask = (
            (np.abs(pts_rot[:, 0]) <= half_size[0]) &
            (np.abs(pts_rot[:, 1]) <= half_size[1]) &
            (np.abs(pts_rot[:, 2]) <= half_size[2])
        )
        return points[mask], intensity[mask], mask

    @staticmethod
    def distortImagePoint(pixels:np.ndarray,
                          camera_intrinsics: np.ndarray,
                          distort: np.ndarray) -> np.ndarray:
        """
        Distort Image Point funciton.
        - Input:
            + "pixels": (N,3) Numpy array with the pixel projection of a point cloud onto an Image.
            + "camera_intrinsics": (3,3) NumPy array of the camera intrinsics parameters. Column stored.
            + "distortion_coef": (1,4) NumPy array of the distortion coefficients
        - Output: 
            * points (np.ndarray): (N,3) Numpy array of distorted point pixel coordinates. 
        """
        # Undo the projection
        xn=(pixels[:,0] - camera_intrinsics[0,2])/camera_intrinsics[0,0]
        yn=(pixels[:,1] - camera_intrinsics[1,2])/camera_intrinsics[1,1]

        # Define variables
        x2 = xn**2
        y2 = yn**2
        rr2=x2 + y2
        rr4=rr2**2
        rr6=rr2**3
        xyn=xn*yn
        
        # Apply distortion
        val=1+distort[0]*rr2 + distort[1]*rr4 + distort[4]*rr6
        x_dist=xn*val + 2*distort[2]*xyn + distort[3]*(rr2 + 2*x2)
        y_dist=yn*val + distort[2]*(rr2 + 2*y2)  + 2*distort[3]*xyn

        # Reproject to pixel coordinates
        u=x_dist * camera_intrinsics[0,0] + camera_intrinsics[0,2]
        v=y_dist * camera_intrinsics[1,1] + camera_intrinsics[1,2]  
    
        pixels[:,0]=u
        pixels[:,1]=v
    
        return pixels
    
    @staticmethod
    def local_to_geodetic(ego_lon, ego_lat, object_coords, heading_deg, ego_alt=0.0):
        """
        Convert local vehicle coordinates to global lat/lon, taking into account ego heading.

        Parameters:
            ego_lon (float): Longitude of the ego vehicle.
            ego_lat (float): Latitude of the ego vehicle.
            object_coords (list of tuples): List of (x, y, z) positions in ego frame.
            heading_deg (float): Heading of the ego vehicle (degrees, 0° = north, clockwise).
            ego_alt (float): Altitude of the ego vehicle (default = 0).

        Returns:
            list of tuples: Each tuple is (lon, lat) in global coordinates.
        """
        try:
            from pyproj import CRS, Transformer
        except ImportError as e:
            raise ImportError(
                "The 'pyproj' package is required to use this function. "
                "Please install it with `pip install pyproj`."
            ) from e
        
        # Coordinate systems
        geodetic_crs = CRS.from_epsg(4979)  # WGS84 with height
        ecef_crs = CRS.from_epsg(4978)      # ECEF

        geo_to_ecef = Transformer.from_crs(geodetic_crs, ecef_crs, always_xy=True)
        ecef_to_geo = Transformer.from_crs(ecef_crs, geodetic_crs, always_xy=True)

        # Ego position in ECEF
        x0, y0, z0 = geo_to_ecef.transform(ego_lon, ego_lat, ego_alt)
        ego_ecef = np.array([x0, y0, z0])

        # ENU to ECEF rotation matrix
        def enu_to_ecef_matrix(lat, lon):
            lat = np.radians(lat)
            lon = np.radians(lon)
            return np.array([
                [-np.sin(lon), -np.sin(lat)*np.cos(lon),  np.cos(lat)*np.cos(lon)],
                [ np.cos(lon), -np.sin(lat)*np.sin(lon),  np.cos(lat)*np.sin(lon)],
                [0,             np.cos(lat),              np.sin(lat)]
            ])

        R_enu_to_ecef = enu_to_ecef_matrix(ego_lat, ego_lon)

        # Heading rotation matrix (local frame to ENU)
        heading_rad = -(np.radians(heading_deg) - np.pi / 2)
        R_heading = np.array([
            [ np.cos(heading_rad), -np.sin(heading_rad), 0],
            [ np.sin(heading_rad),  np.cos(heading_rad), 0],
            [ 0,                    0,                   1]
        ])

        global_coords = []
        for x_local, y_local, z_local in object_coords:
            local_vec = np.array([x_local, y_local, z_local])
            enu = R_heading @ local_vec  # Rotate local frame into ENU
            ecef_vec = ego_ecef + R_enu_to_ecef @ enu  # Convert to ECEF
            lon_obj, lat_obj, _ = ecef_to_geo.transform(*ecef_vec)
            global_coords.append((lat_obj, lon_obj))

        return global_coords

class LidarPointCloud(PointCloud):
    def __init__(self,
                 data_path: pathlib.Path = None,
                 format: str = 'USEFUL',
                 points: np.ndarray = None,
                 intensity: np.ndarray = None,
                 color: np.ndarray = None) -> None:
        """
        Initializes a LidarPointCloud object either by loading data from a file or from raw arrays.

        Supported formats include standard autonomous driving datasets (KITTI, nuScenes) and 
        custom formats. If loading from a file, it parses points, intensity, and color 
        based on the specified format.

        Args:
            data_path: Path to the point cloud file.
            format: Input file format. Supported: 'pcd', 'USEFUL', 'Normalized', 
                'nuScenes', 'KITTI', 'Box'. Defaults to 'USEFUL'.
            points: Optional (N, 3) numpy array to initialize points directly.
            intensity: Optional (N,) numpy array with intensity values.
            color: Optional (N, 3) numpy array with RGB values.

        Raises:
            AssertionError: If the lengths of points, intensity, or color arrays do not match.
        """
        
        self.color = None
        self.intensity = None
        if data_path is not None:
            self.data_path = pathlib.Path(data_path)
            
            if format == 'pcd':
                point_cloud = o3d.io.read_point_cloud(str(data_path))
                self.point_cloud = point_cloud
                self.points = np.asarray(point_cloud.points)
            elif format == 'USEFUL':
                self.points, self.intensity, self.color = self.__readUsefulPC__(str(data_path))
            elif format == 'Normalized':
                self.points, self.intensity = self.__readUsefulNormalizedPC__(str(data_path))
            elif format == 'nuScenes':
                self.points, self.intensity = self.__readNuScenesPCP__(str(data_path))
            elif format in ['KITTI', 'kitti']:
                self.points, self.intensity = self.__readKittiPC__(str(data_path))
            elif format == 'Box':
                self.points = np.fromfile(str(data_path), dtype=np.float32).reshape(-1, 4)[:,:3]
                self.intensity = np.fromfile(str(data_path), dtype=np.float32).reshape(-1, 4)[:,3]
            else:
                print(f'Format "{format}" not supported')
        elif points is not None:
            self.data_path = None
            self.points = points
        if intensity is not None:
            assert len(self.points) ==  len(intensity), f'Points ({len(self.points)}) and intensity ({len(intensity)}) must have the same length'
            self.intensity = intensity
        if color is not None:
            assert len(self.points) ==  len(color), f'Points ({len(self.points)}) and color ({len(color)}) must have the same length'
            self.color = color
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(self.points)

    def copy(self) -> 'LidarPointCloud':
        """
        Creates a copy of itself.
        """
        return copy.deepcopy(self)
    
    @staticmethod
    def __readUsefulPC__(data_path: str):
        format_string = "=fffHI" # f=float(4), H=uint16(2), I=uint32(4), = --> no padding for H
        data_size = struct.calcsize(format_string)  
        data = []  

        with open(data_path, 'rb') as file:
            while True:
                line = file.readline().decode().strip()
                if line.startswith("DATA binary"):
                    break

            # Read binary data
            while True:
                binary_data = file.read(data_size)
                if not binary_data:
                    break  # End of file reached

                # Unpack the binary data
                unpacked_data = struct.unpack(format_string, binary_data)
                data.append(unpacked_data)

        # Convert collected data to numpy array
        data = np.array(data, dtype=np.float32)

        # Split into points, intensity, and RGB
        points = data[:, :3]
        intensity = data[:, 3]
        rgb = [(0, 0, 0) for rgb_i in data[:, 4]] #[(rgb_i >> 16 & 0xFF, rgb_i >> 8 & 0xFF, rgb_i & 0xFF) for rgb_i in data[:, 4]]
        return points, intensity, rgb
    
    def save_pcd(self,
                 save_path: str,
                 format: str = 'USEFUL') -> None:
        """
        Saves the point cloud to a PCD file.
        """
        assert format in ['USEFUL', 'Box'], f'Format {format} not supported'
        if format == 'USEFUL':
            with open(save_path, 'wb') as file:
                header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity rgb
SIZE 4 4 4 2 4
TYPE F F F U U
COUNT 1 1 1 1 1
WIDTH {len(self)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(self)}
DATA binary
"""
                file.write(header.encode("ascii"))
                format_string = "=fffHI"
                for i in range(len(self)):
                    x = self.points[i,0]
                    y = self.points[i,1]
                    z = self.points[i,2]
                    intensity = int(self.intensity[i])
                    color = 0
                    packed = struct.pack(format_string, x, y, z, intensity, color)
                    file.write(packed)
            print(f"Finished writing pcd into {save_path}")
        elif format == 'Box':
            xyzi = np.hstack((self.points, self.intensity.reshape(-1,1)))
            xyzi.astype(np.float32).tofile(save_path)
            print(f"Finished writing bin into {save_path}")

    @staticmethod
    def __readUsefulNormalizedPC__(data_path: str):
        """
        Reads a PCD file with XYZI values in binary format.

        Args:
            data_path (str): Path to the PCD file.

        Returns:
            tuple: (points, intensity) where:
                - points is an Nx3 NumPy array (X, Y, Z).
                - intensity is an N-element NumPy array.
        """
        format_string = "=ffff"  # Four float32 values (X, Y, Z, Intensity)
        data_size = struct.calcsize(format_string)  # Should be 16 bytes

        with open(data_path, 'rb') as file:
            # Read the header and skip until "DATA binary"
            while True:
                line = file.readline().decode('ascii', errors='ignore').strip()
                if line.startswith("DATA binary"):
                    break  # Found the binary data section

            # Read all remaining binary data
            binary_data = file.read()

        # Ensure the binary data length is a multiple of data_size (16 bytes)
        if len(binary_data) % data_size != 0:
            raise ValueError(f"File size ({len(binary_data)} bytes) does not align with expected XYZI format ({data_size} bytes per point).")

        # Convert binary data to NumPy array
        data = np.frombuffer(binary_data, dtype=np.float32).reshape(-1, 4)

        # Extract points and intensity
        points = data[:, :3]  # XYZ
        intensity = data[:, 3]  # Intensity

        return points, intensity


    @staticmethod
    def __readNuScenesPCP__(data_path: str) -> tuple:
        '''
        Reads Beamagine Pointcloud in bin format.
        - Input:
            * data_path (path): path to bin
        - Output:
            + "points": (N,3) NumPy array of point cloud coordinates
            + "intensity": (N,) NumPy array of point cloud intensity values.
        '''
        # Reads binary file
        fromfile = np.fromfile(data_path, dtype = np.float32).reshape(-1,5)
        points = fromfile[:,:3]
        intensity = fromfile[:,3]
        return points, intensity

    @staticmethod
    def __readKittiPC__(data_path: str) -> tuple:
        '''
        Reads Kitti Pointcloud in bin format from file.
        - Input:
            * data_path (path): path to bin
        - Output:
                + "points": (N,3) NumPy array of point cloud coordinates.
                + "intensity": (N,) NumPy array of point cloud intensity values.
        '''
        size_float = 4
        points = []
        intensity_list = []

        with open(data_path, "rb") as f:
            byte = f.read(size_float * 4)
            while byte:
                x, y, z, intensity = struct.unpack("ffff", byte)
                points.append([x, y, z])
                intensity_list.append(intensity)
                byte = f.read(size_float * 4)
        points = np.asarray(points)
        intensity_list = np.asarray(intensity_list)

        return points, intensity_list
    
    def normalizeIntensity(self,
                           max_value: int = np.inf,
                           min_value: int = 0,
                           equalize: bool = False) -> np.ndarray:
        """
        Normalizes point cloud intensity values through clipping and optional histogram equalization.

        The process first clamps the intensity values to a specified range [min_value, max_value].
        Then, it either applies a global linear normalization to [0, 1] or uses OpenCV's 
        histogram equalization to enhance contrast in the intensity distribution.

        Args:
            max_value: Upper threshold for intensity clipping. Values above this are capped.
            min_value: Lower threshold for intensity clipping. Values below this are floored.
            equalize: If True, performs histogram equalization (useful for low-contrast data).
                     If False, performs simple linear normalization to the [0, 1] range.

        Returns:
            np.ndarray: The updated intensity array with normalized values.
        """
        self.intensity[self.intensity > max_value] = max_value
        self.intensity[self.intensity < min_value] = min_value
        if equalize:
            # Normalize the array to the [0, 255] range
            normalized_array = cv2.normalize(self.intensity, None, 0, 255, cv2.NORM_MINMAX)

            # Apply histogram equalization
            intensity_array = cv2.equalizeHist(normalized_array.astype(np.uint8))
            self.intensity = intensity_array.reshape(self.intensity.shape)
        else:
            self.intensity = self.intensity / np.max(self.intensity)
        return self.intensity
        
    def transportImageColorToPointCloud(self,
                                        image: np.ndarray,
                                        camera_intrinsics: np.ndarray,
                                        extrinsics: np.ndarray,
                                        distortion_vector: np.ndarray = None,
                                        invert_color: bool = True,
                                        visualization: bool = True) -> np.ndarray:
        """
        Maps RGB/BGR colors from a 2D image onto the 3D points of the point cloud.

        This method projects the 3D points to the image plane, samples the color at each 
        pixel location, and assigns it to the corresponding point. Points falling outside 
        the image boundaries or behind the camera are assigned a black color [0, 0, 0].

        Args:
            image: Input image array (H, W, 3).
            camera_intrinsics: 3x3 camera calibration matrix.
            extrinsics: 4x4 transformation matrix from Lidar/World to Camera frame.
            distortion_vector: Optional lens distortion coefficients.
            invert_color: If True, swaps the first and third color channels (e.g., BGR to RGB).
            visualization: If True, opens an Open3D window to display the colored point cloud.

        Returns:
            np.ndarray: An (N, 3) array containing the mapped colors for each point, 
                        normalized to the [0, 1] range.
        """
        
        pixels = self.project_points_to_image(camera_intrinsics= camera_intrinsics,
                                              extrinsic= extrinsics,
                                              distortion_vector=distortion_vector)
        
        h, w = image.shape[0], image.shape[1]
        colors = np.zeros_like(self.points)
        for i in range(len(pixels[:,0])):
            if pixels[i,0] > 0 and pixels[i,0] < w and pixels[i,1] > 0 and pixels[i,1] < h and pixels[i,2] > 0:
                    colors[i,:] = image[int(pixels[i,1]), int(pixels[i,0])] / 255

        if invert_color:
            colors[:,[2,0]] = colors[:,[0,2]] 
        if visualization:
            pointcloud = o3d.geometry.PointCloud()
            pointcloud.points = o3d.utility.Vector3dVector(self.points)
            pointcloud.colors = o3d.utility.Vector3dVector(colors / 255)
            o3d.visualization.draw_geometries([pointcloud])
        return colors
    
    def saveLidarPointcloud(self,
                            path: str,
                            format: str = 'Beamagine',
                            to_mm: bool = False) -> None:
        """
        Saves the point cloud to a file in a specified format.

        Currently supports the PCD (Point Cloud Data) v0.7 format in binary mode, 
        storing spatial coordinates (x, y, z) and intensity for each point.

        Args:
            path: Destination file path (should include .pcd extension).
            format: Output file format. Currently only 'XYZI' is supported.
            to_mm: If True, scales coordinates to millimeters (currently not implemented).

        Raises:
            AssertionError: If the requested format is not in the 'available_formats' list.
            AttributeError: If 'self.intensity' is not defined before saving as 'XYZI'.
        """
        available_formats = ['XYZI']
        assert format in available_formats, f'Format {format} not available, try {available_formats}'

        if format in ['XYZI', 'xyzi']:
            num_points = self.points.shape[0]
            data = np.hstack((self.points, self.intensity[:, np.newaxis])).astype(np.float32)
            
            header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA binary
"""
            with open(path, 'wb') as f:
                f.write(header.encode('ascii'))
                f.write(data.tobytes())            
        print(f"Point cloud written to {path}")
        return 
    
    def normals(self,
                radius: float = 0.1,
                max_nn: int = 30,
                k: int = None):
        """
        Estimates and optionally orients surface normals for the point cloud.

        The estimation uses a Hybrid KDTree search, which considers neighbors within a 
        specified radius while capping the total number of neighbors to ensure performance.
        If 'k' is provided, it orients the normals to be consistent using a tangent 
        plane approach, which is critical for surface reconstruction.

        Args:
            radius: Search radius for identifying neighboring points.
            max_nn: Maximum number of neighbors to consider for local plane fitting.
            k: Number of nearest neighbors to use for consistent normal orientation. 
               If None, normals might point in arbitrary directions (inward/outward).

        Returns:
            np.ndarray: An (N, 3) array containing the estimated normal vectors (nx, ny, nz).
        """
        self.point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                          radius=radius,  # Radius to search for neighbors
                                          max_nn=max_nn    # Max number of neighbors to use
                                      ))
        if k is not None:
            self.point_cloud.orient_normals_consistent_tangent_plane(k=k)
        return np.asarray(self.point_cloud.normals)

    
class RadarPointCloud(PointCloud):
    def __init__(self,
                 data_path: pathlib.Path = None,
                 points: np.ndarray = None) -> None:
        """
        Initializes a RadarPointCloud object from a binary file or raw coordinates.

        When loading from 'data_path', it parses a specific binary format (62 bytes per point) 
        containing spatial coordinates and specialized Radar metadata such as Doppler 
        velocity (speed_radial), RCS, and noise levels.

        Args:
            data_path: Path to the binary radar data file. If provided, triggers 
                the internal binary parser.
            points: Optional (N, 3) numpy array to initialize the point cloud 
                geometry directly without metadata.

        Note:
            The internal 'self.data' matrix stores 13 radar-specific attributes 
            (from range and speed to RCS and noise) mapped via class properties.
        """
        if data_path is not None:
            self.data_path = pathlib.Path(data_path)
            data = self.__readRadarPointCloud__(str(data_path))
            self.points = data[:,:3]
            self.data = data[:,3:]
        elif points is not None:
            self.points = points
            self.data = None
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(self.points)


    def __readRadarPointCloud__(self,
                                data_path: pathlib.Path):
        data = np.empty((0,16))
        format_string = "fffffffffffffffH"
        with open(data_path, 'rb') as file:
            # Skip the header lines
            for _ in range(11):
                file.readline()

            # Read binary data
            while True:
                # Read 48 bytes at a time (12 floats and 2 shorts)
                binary_data = file.read(62)
                if not binary_data:
                    break  # End of file reached

                # Unpack the binary data using the format string
                unpacked_data = np.array(struct.unpack(format_string, binary_data))

                # Append the unpacked data to the list
                data = np.vstack((data, unpacked_data))
        return data
    
    @property
    def range(self) -> np.ndarray:
        return self.data[:,0]
    @property
    def speed_radial(self) -> np.ndarray:
        return self.data[:,1]
    @property
    def azimuth(self) -> np.ndarray:
        return self.data[:,2]
    @property
    def elevation(self) -> np.ndarray:
        return self.data[:,3]
    @property
    def variance_range(self) -> np.ndarray:
        return self.data[:,4]
    @property
    def variance_speed(self) -> np.ndarray:
        return self.data[:,5]
    @property
    def variance_azimuth(self) -> np.ndarray:
        return self.data[:,6]
    @property
    def variance_elevation(self) -> np.ndarray:
        return self.data[:,7]
    @property
    def RCS(self) -> np.ndarray:
        return self.data[:,8]
    @property
    def false_alarm_probability(self) -> np.ndarray:
        return self.data[:,9]
    @property
    def power(self) -> np.ndarray:
        return self.data[:,10]
    @property
    def noise(self) -> np.ndarray:
        return self.data[:,11]
    @property
    def peak_index(self) -> np.ndarray:
        return self.data[:,12]
    
    def filter_distance(self,
                        min: float = 0,
                        max: float = np.inf,
                        mode: str = 'Euclidean') -> None:
        
        assert max > min, 'Max distance should be larger than Min distance.'

        if mode == 'Euclidean':
            distances = self.distances()
        if self.data is not None:
            self.data = self.data[np.logical_and(distances > min , distances < max), :]
        self.points = self.points[np.logical_and(distances > min , distances < max), :]
        self.point_cloud.points = o3d.utility.Vector3dVector(self.points)
        if np.asarray(self.point_cloud.colors).size != 0: 
            colors = np.asarray(self.point_cloud.colors)
            colors = colors[np.logical_and(distances > min, distances < max), :]
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

class Image:
    def __repr__(self) -> str:
        repr_str = f'Data path: {self.data_path}\nShape: {self.shape}\nModality: {self.modality}\nImage: {self.image}'
        return repr_str 
    @property
    def shape(self):
        return self.image.shape
    
    @staticmethod
    def draw_projected_boxes3d(image: np.ndarray,
                               qs: np.ndarray,
                               orientation: bool = True,
                               color: tuple = (0, 0, 255),
                               thickness: int = 1):
        """
        Draw 3d bounding box in image. 
        qs: (8,2) array of vertices in the following order:
        1------- 0
        /|      / |
        2-------3 .
        | |     | |
        . 5-------4
        |/       |/
        6--------7
        """

        img = image.copy()

        qs = qs.astype(np.int32)
        for k in range(0,4):
            i, j = k, (k+1)%4
            cv2.line(img, (qs[i,0],qs[i,1]),(qs[j,0],qs[j,1]),color, thickness)
            i, j = k+4, (k+1)%4 +4
            cv2.line(img, (qs[i,0],qs[i,1]),(qs[j,0],qs[j,1]),color, thickness)
            i, j = k, k+4
            cv2.line(img, (qs[i,0],qs[i,1]),(qs[j,0],qs[j,1]),color, thickness)
        if orientation:
            center = np.mean(qs, axis=0)
            #center_forward = np.mean(qs[[2, 3, 7, 6],:], axis=0)
            center_forward = np.mean(qs[[0, 1, 5, 4],:], axis=0)
            cv2.line(img,
                 (int(center[0]), int(center[1])),
                 (int(center_forward[0]), int(center_forward[1])),
                 color, thickness)
        return img
        
    @staticmethod
    def draw_projected_boxes2d(image: np.ndarray,
                               pixels: np.ndarray,
                               color: tuple = (0, 0, 255),
                               thickness: int = 10,
                               crop_3d_bbox: bool = True,
                               plot_corners: bool = False):
        img = image.copy()
        x_max =  img.shape[1]
        y_max = img.shape[0]
        if crop_3d_bbox:
            pixels = Box.crop_3d_bbox(pixels,
                                        x_max = x_max,
                                        y_max = y_max) 

        top = int(np.max([np.min(pixels[:,1]),0]))
        left = int(np.max([np.min(pixels[:,0]),0]))
        bottom = int(np.min([np.max(pixels[:,1]), y_max]))
        right = int(np.min([np.max(pixels[:,0]), x_max]))
        
        img = cv2.rectangle(img, (left,top), (right,bottom), color= color, thickness=thickness)
        if plot_corners:
            for i in range(len(pixels)):
                img = cv2.circle(img, (int(pixels[i,0]), int(pixels[i,1])), 15, color, -1)
        return img
    
    def detect_chessboard(self,
                          chessboard: tuple = (6,9),
                          show: bool = False) -> np.ndarray:
        '''
        Detect calibration chessboard in image
        :param: chessboard: chessboard shape (3,3) for RADAR
        :output: corners2: pixel position of saddle points detected
        '''
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        copy = self.image.copy()
        gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard, None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            if show:
                copy = cv2.drawChessboardCorners(copy, chessboard, corners, ret)
                cv2.imshow('Chessboard', copy)
                cv2.waitKey(0)
                cv2.destroyWindow('Chessboard')
            return corners2
        else:
            print(f'Did not find pattern.')
            if show:
                cv2.imshow('Chessboard', copy)
                cv2.waitKey(0)
                cv2.destroyWindow('Chessboard')
            return None


    def render(self,
               window_name: str = 'Image',
               width: int = 640,
               height: int = 480,
               boxes: list = [],
               lidar: 'LidarPointCloud' = None,
               point_size: int = 1,
               radar: 'RadarPointCloud' = None,
               colormap: str = 'jet',
               loop: float = 5, 
               camera_intrinsics: np.ndarray = None,
               extrinsic: np.ndarray = None,
               distortion_vector: np.ndarray = None,
               bbox_2d: bool = False,
               bbox_3d: bool = False,
               plot_corners: bool = False,
               show: bool = True) -> np.ndarray:
        """
        Renders a 2D visualization by fusing camera images with LiDAR, Radar, and bounding boxes.

        This method projects 3D data (LiDAR/Radar points and 3D boxes) onto the 2D image plane 
        using calibration matrices. It supports multiple colormaps for depth/intensity 
        visualization and can draw both 2D and 3D object detection boxes.

        Args:
            window_name: Title of the OpenCV window.
            width: Width of the display window in pixels.
            height: Height of the display window in pixels.
            boxes: List of Box or Box2D objects to be rendered.
            lidar: Optional LidarPointCloud object to project onto the image.
            point_size: Radius of the circles representing LiDAR points.
            radar: Optional RadarPointCloud object to project (rendered as larger dots).
            colormap: Type of color mapping ('jet', 'beamagine', 'intensity', etc.).
            camera_intrinsics: 3x3 camera matrix. Required if 'boxes', 'lidar', or 'radar' are provided.
            extrinsic: 4x4 transformation matrix from sensor frame to camera frame.
            distortion_vector: Optional lens distortion coefficients.
            bbox_2d: If True, draws the 2D projection or crop of the boxes.
            bbox_3d: If True, draws the 3D wireframe of the boxes.
            plot_corners: If True, highlights the individual corners of the 2D boxes.
            show: If True, opens an interactive OpenCV window and waits for a key press.

        Returns:
            np.ndarray: The rendered image (RGB/BGR) with all overlays.

        Raises:
            AssertionError: If boxes are provided without camera calibration parameters.
        """
        
        #assert len(boxes) != 0 and camera_intrinsics == None, 'Fusion parameters are needed to draw 3d boxes'
        copy = self.image.copy()

        if lidar is not None:
            pixels = lidar.project_points_to_image(camera_intrinsics= camera_intrinsics,
                                                   extrinsic=extrinsic,
                                                   distortion_vector=distortion_vector)
            if colormap == 'beamagine':
                colors = lidar.beamagine_color_palette()
                if np.max(copy) < 2:
                    colors = colors / 255
            elif colormap == 'intensity':
                colors = lidar.get_color_for_intensity(max = 2000)
                if np.max(copy) < 2:
                    colors = colors / 255
            else:
                colors = lidar.get_color_for_distance(color_palette=colormap, loop=loop)
                if np.max(copy) < 2:
                    colors = colors / 255
            for i in range(len(pixels[:,0])):
                if pixels[i,2] > 0:
                    copy = cv2.circle(copy, (int(pixels[i, 0]), int(pixels[i, 1])), radius = point_size, color=(colors[i,2], colors[i,1], colors[i,0]), thickness=-1)

        if radar is not None:
            pixels = radar.project_points_to_image(camera_intrinsics=camera_intrinsics,
                                                   extrinsics=extrinsic,
                                                   distortion_vector=distortion_vector)
            
            colors = radar.get_color_for_distance(color_palette=colormap)
            for i in range(len(pixels[:,0])):
                copy = cv2.circle(copy, (int(pixels[i, 0]), int(pixels[i, 1])), radius = 10, color=(colors[i,2], colors[i,1], colors[i,0]), thickness=-1)

        if len(boxes) != 0:
            for box in boxes:
                if isinstance(box, Box):
                    pixels = box.project_corners_to_image(camera_intrinsics=camera_intrinsics,
                                                        extrinsic=extrinsic,
                                                        distortion_vector=distortion_vector)
                    if np.any(pixels[:,2] < 0):
                        continue
                if bbox_2d:
                    if isinstance(box, Box):
                        copy = self.draw_projected_boxes2d(copy, pixels, color=box.color, thickness=2, plot_corners=plot_corners)
                    elif isinstance(box, Box2D):
                        copy = cv2.rectangle(copy,
                                             (int(box.bbox[0]), int(box.bbox[1])), (int(box.bbox[2]), int(box.bbox[3])),
                                             box.color, 2)
                if bbox_3d:
                    copy = self.draw_projected_boxes3d(copy, pixels, color=box.color, orientation= True, thickness=2)
        if show:
            cv2.namedWindow(f'{window_name}', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'{window_name}', width, height)
            cv2.imshow(f'{window_name}', copy)
            cv2.waitKey(0)
            cv2.destroyWindow(f'{window_name}')
        return copy
    
    @staticmethod
    def __readBinImage__(file_path: str,
                         img_shape: tuple):
        """
        Reads a binary file containing float32 image data, reshapes it, and displays it using OpenCV.

        :param file_path: Path to the .bin file
        :param img_shape: Tuple with the original image shape (height, width)
        """
        # Load binary data as float32
        data = np.fromfile(file_path, dtype=np.uint8)

        # Reshape to original image dimensions
        image = data.reshape(img_shape)

        # Normalize for visualization (optional, for better contrast)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = np.uint8(image)  # Convert to 8-bit image for display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        return image

class RGBImage(Image):
    def __init__(self,
                 data_path: pathlib.Path = None,
                 img_shape: tuple = (1544,2064,3), #(1544,2064,3)
                 img: np.ndarray = None) -> None:
        if data_path is not None:
            self.data_path = pathlib.Path(data_path)
            if '.bin' in self.data_path.suffixes:
                assert img_shape is not None, 'img_shape should be specified when loading a .bin image.' 
                self.image = self.__readBinImage__(str(self.data_path), img_shape=img_shape)
            else:
                self.image = cv2.imread(str(self.data_path))
        elif img is not None:
            self.data_path = None
            self.image = np.copy(img)
        self.modality = 'RGB'
class ThermalImage(Image):
    def __init__(self,
                 data_path: pathlib.Path = None,
                 img: np.ndarray = None,
                 colormap: str = 'magma') -> None:
        if data_path is not None:
            self.data_path = pathlib.Path(data_path)
            self.image = cv2.imread(str(self.data_path), cv2.IMREAD_GRAYSCALE)
        elif img is not None:
            self.data_path = None
            self.image = np.copy(img)
        self.modality = 'Thermal'
        self.colormap(colormap)

    def colormap(self, cmap: str = 'magma'):
        """
        Aplica un colormap de Matplotlib a la imatge actual.
        Nota: self.image ha de ser una matriu 2D (grayscale).
        """
        if self.image is None:
            print("Error: No hi ha cap imatge carregada.")
            return

        try:
            get_cmap = plt.get_cmap(cmap)
        except ValueError:
            print(f"Error: colormap '{cmap}' does not exist in Matplotlib.")
            return

        img_min = self.image.min()
        img_max = self.image.max()
        
        if img_max <= img_min:
            norm_img = np.zeros_like(self.image, dtype=np.float32)
        else:
            norm_img = (self.image - img_min) / (img_max - img_min)

        color_mapped = get_cmap(norm_img)

        rgb_img = (color_mapped[:, :, :3] * 255).astype(np.uint8)
        self.image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        

class SWIRImage(ThermalImage):
    def __init__(self,
                 data_path: pathlib.Path = None,
                 img: np.ndarray = None,
                 colormap: str = 'gray') -> None:
        if data_path is not None:
            self.data_path = pathlib.Path(data_path)
            self.image = cv2.imread(str(self.data_path), cv2.IMREAD_GRAYSCALE)
        elif img is not None:
            self.data_path = None
            self.image = np.copy(img)
        self.modality = 'SWIR'
        self.colormap(colormap)

class PolarimetricImage(Image):
    def __init__(self,
                 data_path: pathlib.Path = None,
                 img: np.ndarray = None) -> None:
        if data_path is not None:
            self.data_path = pathlib.Path(data_path)
            self.image = cv2.imread(str(self.data_path), cv2.IMREAD_ANYDEPTH)
        elif img is not None:
            self.data_path = None
            self.image = np.copy(img)
        self.modality = 'Polarimetric'

    def getMode(self,
            mode: str = 'RGB') -> np.ndarray:
        """
        0       1       2       3
        +-------+-------+-------+-------+ 0
        |  90°  |  45°  |  90°  |  45°  |
        |   B   |   B   |   G   |   G   |
        +-------+-------+-------+-------+ 1
        | 135°  |  0°   | 135°  |  0°   |
        |   B   |   B   |   G   |   G   |
        +-------+-------+-------+-------+ 2
        |  90°  |  45°  |  90°  |  45°  |
        |   G   |   G   |   R   |   R   |
        +-------+-------+-------+-------+ 3
        | 135°  |  0°   | 135°  |  0°   |
        |   G   |   G   |   R   |   R   |
        +-------+-------+-------+-------+
        """

        img = np.copy(self.image)

        # B
        B90  = img[::4, ::4]
        if mode == 'B90':
            return B90 / 255
        B45  = img[1::4, ::4]
        if mode == 'B45':
            return B45 / 255
        B135 = img[::4, 1::4]
        if mode == 'B135':
            return B135 / 255
        B0   = img[1::4, 1::4]
        if mode == 'B0':
            return B0 / 255

        # Gb
        Gb90  = img[::4, 2::4]
        if mode == 'Gr90':
            return Gr90 / 255
        Gb45  = img[1::4, 2::4]
        if mode == 'Gr45':
            return Gr45 / 255
        Gb135 = img[::4, 3::4]
        if mode == 'Gr135':
            return Gr135 / 255
        Gb0   = img[1::4, 3::4]
        if mode == 'Gr0':
            return Gr0 / 255
        
        # Gr
        Gr90  = img[2::4, ::4]
        if mode == 'Gb90':
            return Gb90 / 255
        Gr45  = img[2::4, 1::4]
        if mode == 'Gb45':
            return Gb45 / 255
        Gr135 = img[3::4, ::4]
        if mode == 'Gb135':
            return Gb135 / 255
        Gr0   = img[3::4, 1::4]
        if mode == 'Gb0':
            return Gb0 / 255
        
        # R
        R90  = img[2::4, 2::4]
        if mode == 'R90':
            return R90 / 255
        R45  = img[3::4, 2::4]
        if mode == 'R45':
            return R45 / 255
        R135 = img[2::4, 3::4]
        if mode == 'R135':
            return R135 / 255
        R0   = img[3::4, 3::4]
        if mode == 'R0':
            return R0 / 255
        
        # RGB

        I0_raw = np.empty((R0.shape[0] * 2, R0.shape[1] * 2), dtype=np.uint8)
        I0_raw[::2, ::2] = B0
        I0_raw[1::2, ::2] = Gb0
        I0_raw[::2, 1::2] = Gr0
        I0_raw[1::2, 1::2] = R0
        I0 = cv2.cvtColor(I0_raw, cv2.COLOR_BAYER_BG2BGR) / 255
        if mode == 'RGB0':
            return I0
        
        I90_raw = np.empty((R90.shape[0] * 2, R90.shape[1] * 2), dtype=np.uint8)
        I90_raw[::2, ::2] = B90
        I90_raw[1::2, ::2] = Gb90
        I90_raw[::2, 1::2] = Gr90
        I90_raw[1::2, 1::2] = R90
        I90 = cv2.cvtColor(I90_raw, cv2.COLOR_BAYER_BG2BGR) / 255
        if mode == 'RGB90':
            return I90
        
        I45_raw = np.empty((R45.shape[0] * 2, R45.shape[1] * 2), dtype=np.uint8)
        I45_raw[::2, ::2] = B45
        I45_raw[1::2, ::2] = Gb45
        I45_raw[::2, 1::2] = Gr45
        I45_raw[1::2, 1::2] = R45
        I45 = cv2.cvtColor(I45_raw, cv2.COLOR_BAYER_BG2BGR) / 255
        if mode == 'RGB45':
            return I45
        
        I135_raw = np.empty((R135.shape[0] * 2, R135.shape[1] * 2), dtype=np.uint8)
        I135_raw[::2, ::2] = B135
        I135_raw[1::2, ::2] = Gb135
        I135_raw[::2, 1::2] = Gr135
        I135_raw[1::2, 1::2] = R135
        I135 = cv2.cvtColor(I135_raw, cv2.COLOR_BAYER_BG2BGR) / 255
        if mode == 'RGB135':
            return I135
        
        if mode == 'RGB':
            RGB = (I0 + I90) / 2 
            return RGB
        
        if mode == 'DOLP':
            S0 = I0 + I90
            S1 = I0 - I90
            S2 = I45 - I135
            
            dolp = np.sqrt(S1**2 + S2**2) / (S0 + 1e-6)
            return dolp
        if mode == 'AOLP':
            S1 = I0 - I90
            S2 = I45 - I135
            aolp = 0.5 * np.arctan2(S2, S1)  # In radians
            return aolp

class Box:
    """ Simple data class representing a 3d box including label, score and velocity. """

    def __init__(self,
                 center: list,
                 size: list,
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None,
                 instance_token: str = None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """

        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3

        if isinstance(orientation, Quaternion):
            # Already a quaternion
            self.orientation = orientation
            self.yaw = self.yaw_from_quaternion()

        elif isinstance(orientation, (list, tuple, np.ndarray)) and len(orientation) == 4:
            # Orientation given as [w, x, y, z]
            self.orientation = Quaternion(orientation)
            self.yaw = self.yaw_from_quaternion()

        else:
            # Orientation is a scalar yaw angle
            self.orientation = Quaternion(axis=[0, 0, 1], radians=float(orientation))
            self.yaw = float(orientation)
            
            
        self.center = np.array(center)
        self.wlh = np.array(size)
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token
        self.instance_token = instance_token
        self.color = Box.get_box_color(self.label) if self.label is not None else None

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel
    
    def __repr__(self):
        repr_str = f'Token: {self.token}\n' \
                   f'Instance token: {self.instance_token}\n' \
                   f'Label: {self.label}\n' \
                   f'Color: {self.color}\n' \
                   f'Name: {self.name}\n' \
                   f'Score: {self.score:.2f}\n' \
                   f'XYZ: [{self.center[0]:.2f}, {self.center[1]:.2f}, {self.center[2]:.2f}]\n' \
                   f'WLH:[{self.wlh[0]:.2f}, {self.wlh[1]:.2f}, {self.wlh[2]:.2f}]\n' \
                   f'Rotation axis: [{self.orientation.axis[0]:.2f}, {self.orientation.axis[1]:.2f}, {self.orientation.axis[2]:.2f}]\n' \
                   f'Rotation Angle: {self.orientation.degrees:.2f} (degrees), {self.orientation.radians:.2f} (radians)\n' \
                   f'Velocity: [{self.velocity[0]:.2f}, {self.velocity[1]:.2f}, {self.velocity[2]:.2f}]\n' 

        return repr_str
    
    def copy(self) -> 'Box':
        """
        Creates a copy of itself.
        """
        return copy.deepcopy(self)
    
    def as_list(self) -> list:
        """
        Returns the box as a list.
        :return: <list>. [x, y, z, w, l, h, angle, label, score].
        """
        return [self.center[0], self.center[1], self.center[2],
                self.wlh[0], self.wlh[1], self.wlh[2],
                self.orientation.radians,
                self.label, self.score]
    def yaw_from_quaternion(self):
        """
        Compute yaw (rotation around Z) from a quaternion in (w, x, y, z) format.

        Parameters
        ----------
        q : array-like of shape (4,)
            Quaternion [w, x, y, z].

        Returns
        -------
        yaw : float
            Yaw angle in radians (range -pi to pi).
        """
        w, x, y, z = self.orientation.elements

        # Formula for yaw (Z-axis rotation) from quaternion
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw
    
    @staticmethod
    def get_useful_label_dict(format: str = 'Object Detection') -> dict:
        od = ['Object Detection', 'OD', 'od', 'object detection']
        if format in od:
            return {
                'pedestrian': 0,
                'vehicle.car': 1,
                'vehicle.motorcycle': 2,
                'vehicle.construction': 3,
                'vehicle.bus': 4,
                'vehicle.truck': 5,
                'sign': 6,
                'bicycle': 7,
                'personal_mobility': 8,
                'animal': 9
            }
        
    
    @staticmethod
    def get_box_color(label: int) -> tuple:
        if type(label) != int:
            label = 0
        color_mapping = {
            0: (0, 0, 255),        # Red
            1: (255, 0, 0),        # Blue
            2: (0, 255, 0),        # Green
            3: (0, 255, 255),      # Yellow
            4: (0, 165, 255),      # Orange
            5: (128, 0, 128),      # Purple
            6: (255, 255, 0),      # Cyan
            7: (255, 0, 255),      # Magenta
            8: (0, 255, 128),      # Lime
            9: (203, 192, 255),    # Pink
            10: (128, 128, 0),     # Teal
            11: (42, 42, 165),     # Brown
            12: (0, 0, 128),       # Maroon
            13: (0, 128, 128),     # Olive
            14: (128, 0, 0),       # Navy
            15: (255, 140, 0),     # Dark Orange
            16: (75, 0, 130),      # Indigo
            17: (102, 205, 170),   # Medium Aquamarine
        }
        return color_mapping[label % 18]

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3,3>. The box's rotation matrix.
        """
        return self.orientation.rotation_matrix
    
    def translate(self,
                  x:np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3,1>. Translation in x, y, z direction.
        """
        self.center += x
        return self
    
    def rotate(self,
               quaternion: Quaternion) -> None:
        """
        Define a rotation as a quaternion: cos(alpha/2) + sin(alpha/2)(ai+bj+ck). This will rotate the box an angle alpha around the direction (a,b,c).
        Rotates box.
        :param quaternion: Rotation to apply.
        """


        #self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)
        new_orientation = Quaternion(matrix = quaternion.rotation_matrix @ self.rotation_matrix)
        self.orientation = new_orientation

        return self
    
    def corners(self,
                wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 8, 3>. 
             1------- 0
            /|      / |
            2-------3 .
            | |     | |
            . 5-------4
            |/       |/
            6--------7
        """  
        w, l, h = self.wlh * wlh_factor

        # 3D BBox corners. (Convention: {x:forward, y: left, z: up})
        #                              0   1   2   3   4   5   6   7
        x_corners = w / 2 * np.array([ 1,  1, -1, -1,  1,  1, -1, -1]) 
        y_corners = l / 2 * np.array([-1,  1,  1, -1, -1,  1,  1, -1])     
        z_corners = h / 2 * np.array([ 1,  1,  1,  1, -1, -1, -1, -1]) 
        corners = np.vstack((x_corners, y_corners, z_corners))
        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners.T
    
    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 4, 3>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, 4:]
    
    def get_OBB(self) -> o3d.geometry.OrientedBoundingBox:
        """
        Returns the OrientedBoundingBox object.
        """
        OBB = o3d.geometry.OrientedBoundingBox(self.center, self.rotation_matrix, self.wlh)
        OBB.color = self.color if self.color is not None else np.array((1,0,0))
        return OBB
    
    def get_LineSet(self) -> o3d.geometry.LineSet:
        """
        Returns the LineSet object for the orientation of the bounding box.
        """
        points = np.empty((0,3))
        norm = self.rotation_matrix[:,0]
        points = np.vstack((points, self.center, self.center + self.wlh[0] / 2 * norm))
        v3dv_points = o3d.utility.Vector3dVector(points)
        lines = o3d.utility.Vector2iVector([[0,1]])
        return o3d.geometry.LineSet(points = v3dv_points, lines = lines)
    
    def project_corners_to_image(self,
                                camera_intrinsics: np.ndarray,
                                extrinsic: np.ndarray = np.eye(4),
                                distortion_vector: np.ndarray = None) -> np.ndarray:
        """
        Projects 3D bounding box corners onto the image plane using camera intrinsic and extrinsic parameters.

        :param camera_intrinsics: <np.ndarray: 3x3>. Camera intrinsic matrix representing the internal camera parameters.
                                  Typically contains focal lengths and optical centers.
        :param extrinsic: <np.ndarray: 4x4>. Extrinsic matrix representing the camera's extrinsic parameters.
                          Contains the rotation and translation of the camera with respect to the world coordinates.
                          Defaults to the identity matrix.
        :return: <np.ndarray: Nx2>. Array containing the projected corner points in pixel coordinates (x, y) on the image plane.

        Note: This function assumes the corners are in the object coordinate system. Ensure that the corners are properly transformed
        to this coordinate system before passing them to this function.
        """
        assert camera_intrinsics.shape == (3,3), f'Shape {camera_intrinsics.shape}, does not match expected shape (3,3) for camera_intrinsics'
        assert extrinsic.shape == (4,4), f'Shape {extrinsic.shape}, does not match expected shape (4,4) for extrinsic'

        corners = self.corners()
        #corners = order_box_points(corners)
        homogeneous_points = np.hstack((corners, np.ones((corners.shape[0], 1))))
        transformed_points = (extrinsic @ homogeneous_points.T).T

        points = transformed_points[:,:3]

        projected_points = (camera_intrinsics @ points.T).T

        pixels = np.empty_like(projected_points[:,:2])
        pixels[:,0] = projected_points[:,0] / projected_points[:,2]
        pixels[:,1] = projected_points[:,1] / projected_points[:,2]

        if distortion_vector is not None:
            assert len(distortion_vector) == 5, f'The number of components in distortion_vector ({len(distortion_vector)}) is not what is expected (5)'
            pixels = PointCloud.distortImagePoint(pixels, camera_intrinsics, distortion_vector)
            pixels = np.hstack([pixels, projected_points[:,2:3]])
        return pixels
    
    
    def get_points_within_bounding_box(self,
                                       points: np.ndarray):
        """
        Filters and returns points that are contained within the oriented bounding box (OBB).

        This method calculates the Oriented Bounding Box of the current instance and 
        uses it to perform a spatial query on the provided set of points. This is 
        useful for cropping point clouds or identifying points belonging to a 
        specific detected object.

        Args:
            points: (N, 3) numpy array representing the point cloud to be filtered.

        Returns:
            tuple: A tuple containing:
                - inliers (np.ndarray): (M, 3) array of points located inside the OBB.
                - ids (list): List of indices of the inlier points relative to the input array.

        Note:
            The method uses Open3D's 'get_point_indices_within_bounding_box' which 
            is computationally optimized for 3D spatial checks.
        """
        OBB = self.get_OBB()
        ids = OBB.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points))
        inliers = points[ids, :]
        return inliers, ids
    
    @staticmethod
    def crop_3d_bbox(corners: np.ndarray,
                 x_max: int,
                 y_max: int):
        """
        Clamps 3D bounding box projected corners to the image boundaries.

        If a corner is outside the image frame, this method calculates the intersection 
        point between the edge (the line connecting the out-of-bounds corner to an 
        in-bounds neighbor) and the image boundary. This ensures that the box can still 
        be drawn correctly even when partially occluded by the camera frame.

        Args:
            corners: (8, 2) array of projected 2D coordinates of the box corners.
            x_max: Maximum horizontal boundary (usually image width).
            y_max: Maximum vertical boundary (usually image height).

        Returns:
            np.ndarray: Updated (8, 2) array with corners adjusted to fit within [0, x_max] 
                        and [0, y_max].
        """
        new_corners = corners.copy()
        # Note: the connectivity here defines all neighbors of each corner.
        connections = {
            0: [1, 4, 3],
            1: [0, 2, 5],
            2: [1, 3, 6],
            3: [0, 2, 7],
            4: [0, 5, 7],
            5: [1, 4, 6],
            6: [2, 5, 7],
            7: [3, 4, 6]
        }

        for i in range(len(corners)):
            # Check if the current corner is out-of-bounds
            out_x = corners[i, 0] > x_max or corners[i, 0] < 0
            out_y = corners[i, 1] > y_max or corners[i, 1] < 0

            if out_x or out_y:
                # Try each connected neighbor until one is (at least partially) in–bounds.
                for j in connections[i]:
                    if (0 <= corners[j, 0] <= x_max) and (0 <= corners[j, 1] <= y_max):
                        # Create a line from the neighbor to the current corner.
                        line = Line(corners[j, :], corners[i, :])
                        
                        # Depending on which coordinate is out-of-bounds, compute the intersection.
                        # If x is out-of-bounds, compute the intersection at x = x_max or 0.
                        if out_x:
                            if corners[i, 0] > x_max:
                                new_x = x_max
                            else:  # if it's less than 0
                                new_x = 0
                            new_y = line.get_y_from_x(new_x)
                            # If new_y is None (vertical line), we might choose the neighbor's y.
                            if new_y is None:
                                new_y = corners[j, 1]
                            new_corners[i] = [new_x, new_y]
                        
                        # Else, if y is out-of-bounds, compute the intersection at y = y_max or 0.
                        if out_y:
                            if corners[i, 1] > y_max:
                                new_y = y_max
                            else:  # if it's less than 0
                                new_y = 0
                            new_x = line.get_x_from_y(new_y)
                            new_corners[i] = [new_x, new_y]
                        
                        # Once fixed with one neighbor, exit the neighbor loop.
                        break
        return new_corners
    def get_projected_center(self,
                             camera_intrinsics: np.ndarray,
                             extrinsic: np.ndarray = np.eye(4),
                             distortion_vector: np.ndarray = None) -> np.ndarray:
        corners = self.project_corners_to_image(camera_intrinsics=camera_intrinsics,
                                                extrinsic=extrinsic,
                                                distortion_vector=distortion_vector)
        x = 0
        y = 0
        for i in range(len(corners)):
            x += corners[i, 0]
            y += corners[i, 1]
        x /= 8
        y /= 8
        return np.array([x, y])
class Line:
    def __init__(self, p1, p2):
        self.m, self.n = self.get_m_n(p1, p2)

    def get_m_n(self, p1, p2):
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        dx = x2 - x1
        # Avoid division by zero (vertical line)
        if np.abs(dx) < 1e-8:
            return None, x1  
        m = (y2 - y1) / dx
        n = y1 - m * x1
        return m, n

    def get_x_from_y(self, y):
        if self.m is None:
            # Vertical line: x is constant
            return self.n
        return (y - self.n) / self.m

    def get_y_from_x(self, x):
        if self.m is None:
            return None
        return self.m * x + self.n
    
class Box2D:
    def __init__(self,
                 bbox: list,
                 label: int = np.nan,
                 score: float = np.nan,
                 name: str = None,
                 token: str = None,
                 instance_token: str = None,
                 visibility: int = 4):
        """
        :param bbox: (Left, Top, Right, Bottom) corners of the box.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """

        assert not np.any(np.isnan(bbox))
        assert len(bbox) == 4

        self.bbox = bbox
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.name = name
        self.token = token
        self.instance_token = instance_token
        self.visibility = visibility
        self.color = Box.get_box_color(self.label)
        
    def __repr__(self):
        repr_str = f'Bbox: {self.bbox}\n' \
                   f'Instance token: {self.instance_token}\n' \
                   f'Label: {self.label}\n' \
                   f'Color: {self.color}\n' \
                   f'Token: {self.token}\n' \
                   f'Name: {self.name}\n' \
                   f'Score: {self.score:.2f}\n' \

        return repr_str
    def copy(self) -> 'Box2D':
        """
        Creates a copy of itself.
        """
        copy_ = Box2D(self.bbox,
                      self.label,
                      self.score,
                      self.name,
                      self.token,
                      self.instance_token,
                      self.visibility)
        return copy_
    
    @property
    def center(self) -> np.ndarray:
        """
        Returns the center of the box.
        :return: <np.float: 2,1>. The box's center.
        """
        x = (self.bbox[0] + self.bbox[2]) / 2
        y = (self.bbox[1] + self.bbox[3]) / 2
        return np.array([x, y])
    
class CalibUtils:
    @staticmethod
    def get_extrinsics_from_quaternion_and_translation(quaternion: Quaternion,
                                                translation: np.ndarray) -> np.ndarray:
        """
        Returns the extrinsic matrix from a quaternion and a translation vector.
        :param quaternion: <Quaternion>. The rotation of the camera.
        :param translation: <np.float: 3,1>. The translation of the camera.
        :return: <np.float: 4,4>. The extrinsic matrix.
        """
        assert quaternion is not None, 'Quaternion should not be None'
        assert translation is not None, 'Translation should not be None'
        assert len(translation) == 3, f'Translation should be of length 3, but got {len(translation)}'
        
        quaternion = Quaternion(quaternion)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = quaternion.rotation_matrix
        extrinsic[:3, 3] = translation
        return extrinsic