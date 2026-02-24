import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def custom_3d_visualizer(window_name: str = 'Open3D',
                         window_width: int = 1920,
                         window_height: int = 1080,
                         window_left: int = 50,
                         window_top: int = 50,
                         point_size: float = 5.0,
                         line_width: float = 1.0,
                         background_color: tuple = (1,1,1),
                         show_coordinate_frame: bool = False,
                         light_on: bool = True,
                         save_path: str = '',
                         save_ext: str = 'png'):
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name = window_name,
                      width = window_width,
                      height = window_height,
                      left = window_left,
                      top = window_top)
    
    # Set render_option
    vis.get_render_option().background_color = background_color
    vis.get_render_option().show_coordinate_frame = show_coordinate_frame
    vis.get_render_option().point_size = point_size
    vis.get_render_option().light_on = light_on
    vis.get_render_option().line_width = line_width

    # Define movement callbacks
    def rotate_camera_movement(vis):
            ctr = vis.get_view_control()
            ctr.rotate(5.0,0.0)
    def stop_camera_movement(vis):
            ctr = vis.get_view_control()
            ctr.rotate(0.0,0.0)

    # Define key callbacks
    def stop_callback(vis):
        vis.register_animation_callback(stop_camera_movement)

    def rotation_callback(vis):
        vis.register_animation_callback(rotate_camera_movement)

    def random_background_callback(vis):
        background_color = (np.random.rand(1), np.random.rand(1), np.random.rand(1))
        vis.get_render_option().background_color = background_color

    def toggle_bw_background_callback(vis):
        if np.array_equal(vis.get_render_option().background_color, np.array([0,0,0])):
            vis.get_render_option().background_color =  (1,1,1)
        else:
            vis.get_render_option().background_color =  (0,0,0)

    def toggle_axes_callback(vis):
        if vis.get_render_option().show_coordinate_frame == True:
            vis.get_render_option().show_coordinate_frame = False
        else:
            vis.get_render_option().show_coordinate_frame = True
    
    def add_open3d_axis(vis):
        """Add a small 3D axis on Open3D Visualizer"""
        axis = o3d.geometry.LineSet()
        axis.points = o3d.utility.Vector3dVector(np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]))
        axis.lines = o3d.utility.Vector2iVector(np.array([
            [0, 1],
            [0, 2],
            [0, 3]]))
        axis.colors = o3d.utility.Vector3dVector(np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]))
        vis.add_geometry(axis)


    def save_screen_callback(vis):
        file_path = os.path.join(save_path, f'{window_name}_{int(time.time())}.{save_ext}')
        print(f'Image saved at: {file_path}')
        vis.capture_screen_image(file_path, do_render = True)

    def save_depth_callback(vis):
        file_path = os.path.join(save_path, f'{window_name}_{time.time()}.{save_ext}')
        print(f'Image saved at: {file_path}')
        vis.capture_depth_image(file_path, do_render = True)
    
    def increase_point_size_callback(vis):
        point_size = vis.get_render_option().point_size
        vis.get_render_option().point_size = point_size + 2
        vis.update_renderer()

    def decrease_point_size_callback(vis):
        point_size = vis.get_render_option().point_size
        if point_size > 3:
            vis.get_render_option().point_size = point_size - 2
            vis.update_renderer()
        else:
            pass


    # Register key callback function

    vis.register_key_callback(ord('L'), random_background_callback)
    vis.register_key_callback(ord('K'), toggle_bw_background_callback)
    vis.register_key_callback(ord('X'), add_open3d_axis)
    #vis.register_key_callback(ord('X'), toggle_axes_callback)
    vis.register_key_callback(ord('R'), rotation_callback)
    vis.register_key_callback(ord('T'), stop_callback)
    vis.register_key_callback(ord('S'), save_screen_callback)
    vis.register_key_callback(ord('D'), save_depth_callback)
    vis.register_key_callback(ord('M'), increase_point_size_callback)
    vis.register_key_callback(ord('N'), decrease_point_size_callback)
    return vis