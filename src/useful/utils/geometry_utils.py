import numpy as np
import cv2
import folium

def generate_map(points: np.ndarray,
                 file_name: str = None,
                 zoom_start: int = 50,
                 tile_layer: str = 'openstreetmap',
                 color: str = 'red',
                 start_point: bool = True,
                 show_in_browser: bool = False
                 ):
    initial_point = points[0,:]

    m = folium.Map(location=initial_point, zoom_start=zoom_start)

    folium.TileLayer(tile_layer).add_to(m)
    if start_point:
        folium.Marker(location=initial_point, popup='Starting point'). add_to(m)
    folium.PolyLine(locations=points, color = color, weight = 5, opacity = 1).add_to(m)
    if file_name is not None:
        m.save(file_name)
    if show_in_browser:
        m.show_in_browser()
    return

def generate_map_lines(points_list: list,
                    file_name: str = None,
                    zoom_start: int = 50,
                    tile_layer: str = 'openstreetmap',
                    colors: list = ['red'],
                    start_point: bool = True,
                    show_in_browser: bool = False
                    ):
    
    if(len(points_list) != len(colors)):
        assert("points_list and colors list should have the same length.")

    initial_point = points_list[0][0,:]
    m = folium.Map(location=initial_point, zoom_start=zoom_start)

    folium.TileLayer(tile_layer).add_to(m)

    for i in range(len(points_list)):
        if start_point:
            initial_point = points_list[i][0,:]
            folium.Marker(location=initial_point, popup='Starting point'). add_to(m)
        folium.PolyLine(locations=points_list[i], color=colors[i], weight = 5, opacity = 1).add_to(m)

    if file_name is not None:
        m.save(file_name)

    if show_in_browser:
        m.show_in_browser()

    return