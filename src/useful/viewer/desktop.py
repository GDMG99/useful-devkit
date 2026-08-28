"""
Open3D desktop viewer with the same shortcuts as the web viewer.

    python -m useful.viewer.desktop --dataroot data/useful --version v1.1 --scene dia_19

Needs a display (or an X forward); on a headless server use ``useful.viewer``.
Key callbacks follow the VisualizerWithKeyCallback pattern: every state change
rebuilds the colours and re-adds the geometry with ``reset_bounding_box=False``
so the camera stays where you left it.

NOTE: written on a headless machine and not exercised against a window; the
data path (SceneBrowser / ViewerState) is the same one the web viewer runs.
"""
import argparse

import numpy as np
import open3d as o3d

from .core import KEYMAP, MODALITIES, SceneBrowser, ViewerState


class DesktopViewer:
    def __init__(self, browser: SceneBrowser, state: ViewerState = None):
        self.browser = browser
        self.state = state or ViewerState()
        self.vis = None
        self.pcd = o3d.geometry.PointCloud()
        self.extra = []          # boxes / radar geometries currently added
        self._first = True

    # ------------------------------------------------------------ data
    @property
    def scene(self):
        return self.browser.scenes[self.state.scene_idx]

    @property
    def sample_token(self):
        toks = self.browser.sample_tokens(self.scene['token'])
        self.state.sample_idx = max(0, min(self.state.sample_idx, len(toks) - 1))
        return toks[self.state.sample_idx]

    def build(self):
        st = self.state
        tok = self.sample_token
        pts, _ = self.browser.cloud(tok)
        rgb = self.browser.colors(tok, st.color_mode, st.cmap, st.modality, st.max_range)
        keep = np.linalg.norm(pts, axis=1) < st.max_range
        return pts[keep].astype(np.float64), rgb[keep].astype(np.float64) / 255.0

    def extras(self):
        geoms = []
        tok = self.sample_token
        if self.state.show_boxes:
            for b in self.browser.boxes(tok):
                corners = np.asarray(b['corners'])
                edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                         [0, 4], [1, 5], [2, 6], [3, 7]]
                ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(corners),
                                          o3d.utility.Vector2iVector(edges))
                ls.paint_uniform_color(np.asarray(b['color']) / 255.0)
                geoms.append(ls)
        if self.state.show_radar:
            r = self.browser.radar(tok)
            for p in r['points']:
                m = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
                m.translate(p)
                m.paint_uniform_color([1.0, 0.2, 0.2])
                geoms.append(m)
        return geoms

    # ------------------------------------------------------------ render
    def refresh(self):
        pts, cols = self.build()
        self.pcd.points = o3d.utility.Vector3dVector(pts)
        self.pcd.colors = o3d.utility.Vector3dVector(cols)
        if self.vis is None:
            return
        for g in self.extra:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self.extra = self.extras()
        if self._first:
            self.vis.add_geometry(self.pcd, reset_bounding_box=True)
            self._first = False
        else:
            self.vis.remove_geometry(self.pcd, reset_bounding_box=False)
            self.vis.add_geometry(self.pcd, reset_bounding_box=False)
        for g in self.extra:
            self.vis.add_geometry(g, reset_bounding_box=False)
        self.vis.get_render_option().point_size = self.state.point_size
        self.vis.update_renderer()
        info = self.browser.sample_info(self.sample_token)
        print(f"{info['scene_name']} [{info['index'] + 1}/{info['count']}]  {self.state.describe()}")

    # ------------------------------------------------------------ callbacks
    def _step(self, n):
        def cb(vis):
            self.state.sample_idx += n
            self.refresh()
            return True
        return cb

    def _scene(self, n):
        def cb(vis):
            self.state.scene_idx = (self.state.scene_idx + n) % len(self.browser.scenes)
            self.state.sample_idx = 0
            self.refresh()
            return True
        return cb

    def _set_modality(self, m):
        def cb(vis):
            self.state.modality = m
            self.refresh()
            return True
        return cb

    def _cycle(self, fn):
        def cb(vis):
            fn()
            self.refresh()
            return True
        return cb

    def _toggle(self, attr):
        def cb(vis):
            setattr(self.state, attr, not getattr(self.state, attr))
            self.refresh()
            return True
        return cb

    def _psize(self, d):
        def cb(vis):
            self.state.point_size = max(1.0, self.state.point_size + d)
            vis.get_render_option().point_size = self.state.point_size
            vis.update_renderer()
            return True
        return cb

    def _range(self, d):
        def cb(vis):
            self.state.max_range = max(10.0, self.state.max_range + d)
            self.refresh()
            return True
        return cb

    def register(self, vis):
        vis.register_key_callback(ord('N'), self._step(+1))
        vis.register_key_callback(ord('P'), self._step(-1))
        vis.register_key_callback(ord(','), self._scene(-1))
        vis.register_key_callback(ord('.'), self._scene(+1))
        vis.register_key_callback(ord('M'), self._cycle(self.state.cycle_modality))
        for i, m in enumerate(MODALITIES):
            vis.register_key_callback(ord(str(i)), self._set_modality(m))
        vis.register_key_callback(ord('I'), self._cycle(self.state.cycle_color_mode))
        vis.register_key_callback(ord('C'), self._cycle(self.state.cycle_cmap))
        vis.register_key_callback(ord('B'), self._toggle('show_boxes'))
        vis.register_key_callback(ord('R'), self._toggle('show_radar'))
        for k in ('+', '='):
            vis.register_key_callback(ord(k), self._psize(+1.0))
        for k in ('-', '_'):
            vis.register_key_callback(ord(k), self._psize(-1.0))
        vis.register_key_callback(ord('['), self._range(-10.0))
        vis.register_key_callback(ord(']'), self._range(+10.0))
        vis.register_key_callback(ord('H'), lambda v: (print(KEYMAP), True)[1])

    def run(self, width=1400, height=900):
        print(KEYMAP)
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name='USEFUL viewer', width=width, height=height)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.06, 0.06, 0.08])
        opt.point_size = self.state.point_size
        self.vis = vis
        self.refresh()
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0),
                         reset_bounding_box=False)
        self.register(vis)
        vis.run()
        vis.destroy_window()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default='data/useful')
    ap.add_argument('--version', default='v1.1')
    ap.add_argument('--scene', default=None, help='scene name (default: first)')
    ap.add_argument('--sample', type=int, default=0)
    ap.add_argument('--point-size', type=float, default=2.0)
    args = ap.parse_args(argv)
    browser = SceneBrowser(dataroot=args.dataroot, version=args.version)
    state = ViewerState(sample_idx=args.sample, point_size=args.point_size)
    if args.scene:
        names = [s['name'] for s in browser.scenes]
        state.scene_idx = names.index(args.scene)
    DesktopViewer(browser, state).run()


if __name__ == '__main__':
    main()
