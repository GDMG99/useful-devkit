"""
Shared data access + colorization for the USEFUL point-cloud viewers.

Both the web viewer (``useful.viewer.server``) and the Open3D desktop viewer
(``useful.viewer.desktop``) drive the same :class:`SceneBrowser` and
:class:`ViewerState`, so a keyboard shortcut means the same thing in both.
"""
import functools
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import matplotlib

from ..useful import USEFUL
from ..utils.data_classes import (LidarPointCloud, RadarPointCloud, RGBImage,
                                  ThermalImage, SWIRImage, PolarimetricImage, PointCloud)

COLORMAPS = ['jet', 'turbo', 'viridis', 'plasma', 'gray']
COLOR_MODES = ['distance', 'intensity', 'height']
MODALITIES = ['none', 'WIDE_LEFT', 'NARROW', 'WIDE_RIGHT', 'LWIR', 'POLARIMETRIC', 'SWIR']
FUSION_FALLBACK = np.array([70, 70, 78], dtype=np.uint8)   # points outside the camera FOV
INTENSITY_MAX = 2000.0                                     # same clip the devkit renders with
DISTANCE_COLOR_MAX = 60.0                                  # colormap span for 'distance' (m)

KEYMAP = """
 ------------------------------------------------------------------
  USEFUL viewer -- keys
 ------------------------------------------------------------------
  N / P  (-> / <-)   next / previous sample        Shift: +-10
  Space              play / pause
  , / .              previous / next scene
  M  (Shift+M)       cycle camera fusion modality (reverse)
  0..6               fusion: 0 none, 1 WIDE_LEFT, 2 NARROW, 3 WIDE_RIGHT,
                             4 LWIR, 5 POLARIMETRIC, 6 SWIR
  I                  cycle colour source: distance / intensity / height
  C                  cycle colormap: jet / turbo / viridis / plasma / gray
  B                  toggle 3D boxes            R   toggle radar
  V                  toggle camera panel        G   toggle grid / axes
  + / -              point size                 [ / ]  max range -/+ 10 m
  T                  top-down view              Z   reset view
  H / ?              this help
 ------------------------------------------------------------------
"""


@dataclass
class ViewerState:
    """Everything a shortcut can change. Pure data, no rendering."""
    scene_idx: int = 0
    sample_idx: int = 0
    color_mode: str = 'distance'
    modality: str = 'none'
    cmap: str = 'jet'
    point_size: float = 2.0
    max_range: float = 100.0
    show_boxes: bool = True
    show_radar: bool = True
    show_camera: bool = True
    playing: bool = False

    def cycle_color_mode(self, step=1):
        self.color_mode = COLOR_MODES[(COLOR_MODES.index(self.color_mode) + step) % len(COLOR_MODES)]

    def cycle_cmap(self, step=1):
        self.cmap = COLORMAPS[(COLORMAPS.index(self.cmap) + step) % len(COLORMAPS)]

    def cycle_modality(self, step=1):
        self.modality = MODALITIES[(MODALITIES.index(self.modality) + step) % len(MODALITIES)]

    def describe(self):
        fusion = self.modality if self.modality != 'none' else '-'
        return (f'colour={self.color_mode} cmap={self.cmap} fusion={fusion} '
                f'size={self.point_size:g} range<{self.max_range:g}m '
                f'boxes={"on" if self.show_boxes else "off"} radar={"on" if self.show_radar else "off"}')


class SceneBrowser:
    """Cached access to scenes, samples, clouds, images, boxes and radar."""

    def __init__(self, dataroot='data/useful', version='v1.1', verbose=False):
        self.dataroot = dataroot
        self.version = version
        self.db = USEFUL(version=version, dataroot=dataroot, verbose=verbose)
        recs = [(self.db.get('sample', s['first_sample_token'])['timestamp'], s) for s in self.db.scene]
        self.scenes = [s for _, s in sorted(recs, key=lambda r: r[0])]
        self._samples = {}

    # ------------------------------------------------------------ scenes / samples
    def scene_list(self):
        return [{'token': s['token'], 'name': s['name'], 'description': s['description'],
                 'nbr_samples': s['nbr_samples'], 'split': s['split'],
                 'valid_ego_pose': bool(s.get('valid_ego_pose', True))} for s in self.scenes]

    def sample_tokens(self, scene_token):
        if scene_token not in self._samples:
            self._samples[scene_token] = self.db.get_sample_tokens_in_scene(scene_token)
        return self._samples[scene_token]

    def sample_info(self, sample_token):
        s = self.db.get('sample', sample_token)
        scene = self.db.get('scene', s['scene_token'])
        toks = self.sample_tokens(scene['token'])
        ins = self.db.get('ego_pose', s['ego_pose_token']).get('INS', {})
        try:
            speed = float(np.hypot(float(ins['velocity_north_mps']), float(ins['velocity_east_mps'])))
        except (KeyError, ValueError):
            speed = None
        return {'token': sample_token, 'scene_token': scene['token'], 'scene_name': scene['name'],
                'index': toks.index(sample_token), 'count': len(toks), 'timestamp': s['timestamp'],
                'channels': sorted(s['data'].keys()), 'speed_mps': speed,
                'n_anns': len(s.get('anns', []))}

    # ------------------------------------------------------------ data
    @functools.lru_cache(maxsize=64)
    def cloud(self, sample_token):
        """(points float32 (N,3), intensity float32 (N,)) in the LiDAR/ego frame."""
        s = self.db.get('sample', sample_token)
        pc = LidarPointCloud(data_path=self.db.get_sample_data_path(s['data']['LIDAR']), format='USEFUL')
        return np.ascontiguousarray(pc.points, dtype=np.float32), np.ascontiguousarray(pc.intensity, dtype=np.float32)

    @functools.lru_cache(maxsize=64)
    def image(self, sample_token, channel):
        """BGR uint8 image for a camera channel, plus (K, lidar2cam, dist, scale).

        ``scale`` maps the calibration's pixel frame onto this image (the
        polarimetric mosaic is demosaiced to a quarter of the calibrated size).
        """
        s = self.db.get('sample', sample_token)
        if channel not in s['data']:
            return None
        sd_token = s['data'][channel]
        sd = self.db.get('sample_data', sd_token)
        path, _, K, T, dist = self.db.get_sample_data(sd_token, selected_anntokens=False)
        modality = sd['sensor_modality']
        if modality == 'rgb':
            img = RGBImage(data_path=path).image
        elif modality == 'thermal':
            img = ThermalImage(path).image
        elif modality == 'swir':
            img = SWIRImage(path).image
        elif modality == 'polarimetric':
            pol = PolarimetricImage(path).getMode('RGB')
            img = (pol / (np.max(pol) + 1e-6) * 255).astype(np.uint8)
        else:
            return None
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        calib_w = float(sd.get('width') or img.shape[1])
        scale = img.shape[1] / calib_w if calib_w else 1.0
        return img, np.asarray(K, dtype=np.float64), np.asarray(T, dtype=np.float64), \
            (np.asarray(dist, dtype=np.float64) if dist is not None and len(dist) else None), scale

    def boxes(self, sample_token):
        out = []
        for box in self.db.get_boxes(sample_token):
            b, g, r = box.color
            out.append({'token': box.token, 'instance': box.instance_token, 'name': box.name,
                        'label': box.label, 'color': [int(r), int(g), int(b)],
                        'center': box.center.tolist(), 'size': box.wlh.tolist(),
                        'yaw': float(box.yaw), 'corners': box.corners().tolist()})
        return out

    def radar(self, sample_token):
        """Radar points transformed into the LiDAR frame, with radial speed and RCS."""
        s = self.db.get('sample', sample_token)
        pts, vel, rcs, side = [], [], [], []
        for ch in ('RADAR_LEFT', 'RADAR_RIGHT'):
            if ch not in s['data']:
                continue
            path, _, _, T, _ = self.db.get_sample_data(s['data'][ch], selected_anntokens=False)
            rp = RadarPointCloud(data_path=path)
            if len(rp) == 0:
                continue
            rp.transform(T)
            pts.append(np.asarray(rp.points, dtype=np.float32))
            vel.append(np.asarray(rp.speed_radial, dtype=np.float32))
            rcs.append(np.asarray(rp.RCS, dtype=np.float32))
            side += [ch] * len(rp)
        if not pts:
            return {'points': [], 'speed_radial': [], 'rcs': [], 'channel': []}
        return {'points': np.concatenate(pts).tolist(), 'speed_radial': np.concatenate(vel).tolist(),
                'rcs': np.concatenate(rcs).tolist(), 'channel': side}

    # ------------------------------------------------------------ colours
    def colors(self, sample_token, color_mode='distance', cmap='jet', modality='none', max_range=None):
        """uint8 (N,3) RGB colours for the sample's cloud under the given state."""
        points, intensity = self.cloud(sample_token)
        base = scalar_colors(points, intensity, color_mode, cmap, max_range)
        if modality == 'none':
            return base
        data = self.image(sample_token, modality)
        if data is None:
            return base
        img, K, T, dist, scale = data
        u, v, z = project(points, K, T, dist, scale)
        h, w = img.shape[:2]
        inside = (z > 0.5) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        out = np.tile(FUSION_FALLBACK, (len(points), 1))
        bgr = img[v[inside].astype(np.int32), u[inside].astype(np.int32)]
        out[inside] = bgr[:, ::-1]
        return out


def scalar_colors(points, intensity, color_mode='distance', cmap='jet', max_range=None):
    if color_mode == 'intensity':
        scalar = np.clip(intensity / INTENSITY_MAX, 0, 1)
    elif color_mode == 'height':
        z = points[:, 2]
        scalar = np.clip((z + 2.5) / 6.0, 0, 1)             # -2.5 m .. +3.5 m
    else:
        d = np.linalg.norm(points, axis=1)
        top = min(max_range, DISTANCE_COLOR_MAX) if max_range else DISTANCE_COLOR_MAX
        scalar = np.clip(d / top, 0, 1)
    lut = matplotlib.colormaps[cmap](np.linspace(0, 1, 256))[:, :3]
    return (lut[(scalar * 255).astype(np.uint8)] * 255).astype(np.uint8)


def project(points, K, T, dist=None, scale=1.0):
    """Vectorized devkit projection (lidar -> camera pixels). Returns u, v, depth."""
    P = np.hstack([points.astype(np.float64), np.ones((len(points), 1))])
    C = (T @ P.T).T[:, :3]
    z = C[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    pix = np.stack([C[:, 0] / safe, C[:, 1] / safe, z], axis=1)
    pix[:, 0] = pix[:, 0] * K[0, 0] + K[0, 2]
    pix[:, 1] = pix[:, 1] * K[1, 1] + K[1, 2]
    if dist is not None and len(dist) >= 5:
        pix = PointCloud.distortImagePoint(pix, K, dist)
    return pix[:, 0] * scale, pix[:, 1] * scale, z


def overlay_points(img, points, colors, K, T, dist, scale, radius=2, max_range=None):
    """Draw projected points on a copy of the image (BGR)."""
    canvas = img.copy()
    u, v, z = project(points, K, T, dist, scale)
    h, w = canvas.shape[:2]
    ok = (z > 0.5) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if max_range:
        ok &= np.linalg.norm(points, axis=1) < max_range
    ui, vi = u[ok].astype(np.int32), v[ok].astype(np.int32)
    col = colors[ok]
    if radius <= 1:
        canvas[vi, ui] = col[:, ::-1]
    else:
        for x, y, c in zip(ui, vi, col):
            cv2.circle(canvas, (int(x), int(y)), radius, (int(c[2]), int(c[1]), int(c[0])), -1)
    return canvas


def encode_jpeg(img, quality=82, max_width=1400):
    if img.shape[1] > max_width:
        f = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b''
