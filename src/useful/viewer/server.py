"""
Web point-cloud viewer for USEFUL.

Serves a small Three.js app plus a JSON/binary API on a local port. Rendering
happens in the browser, so it works on a headless server; forward the port
with SSH to look at it from your machine::

    python -m useful.viewer --dataroot data/useful --version v1.1 --port 8080
    # on your laptop:
    ssh -N -L 8080:localhost:8080 <user>@<server>   ->  http://localhost:8080
"""
import argparse
import json
import os

import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_from_directory

from .core import (COLORMAPS, COLOR_MODES, MODALITIES, KEYMAP, SceneBrowser,
                   encode_jpeg, overlay_points)

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')


def create_app(browser: SceneBrowser) -> Flask:
    app = Flask(__name__, static_folder=STATIC, static_url_path='/static')

    def _float(name, default):
        v = request.args.get(name)
        try:
            return float(v) if v not in (None, '') else default
        except ValueError:
            return default

    def _sample(token):
        try:
            browser.db.get('sample', token)
        except KeyError:
            abort(404, f'unknown sample {token}')
        return token

    @app.route('/')
    def index():
        return send_from_directory(STATIC, 'index.html')

    @app.route('/api/config')
    def config():
        return jsonify({'colormaps': COLORMAPS, 'color_modes': COLOR_MODES,
                        'modalities': MODALITIES, 'keymap': KEYMAP,
                        'version': browser.version, 'dataroot': os.path.abspath(browser.dataroot)})

    @app.route('/api/scenes')
    def scenes():
        return jsonify(browser.scene_list())

    @app.route('/api/scene/<token>')
    def scene(token):
        try:
            rec = browser.db.get('scene', token)
        except KeyError:
            abort(404)
        toks = browser.sample_tokens(token)
        ts = [browser.db.get('sample', t)['timestamp'] for t in toks]
        return jsonify({'token': token, 'name': rec['name'], 'samples': toks, 'timestamps': ts})

    @app.route('/api/sample/<token>/info')
    def info(token):
        return jsonify(browser.sample_info(_sample(token)))

    @app.route('/api/sample/<token>/points')
    def points(token):
        pts, inten = browser.cloud(_sample(token))
        payload = np.ascontiguousarray(pts, dtype='<f4').tobytes()
        return Response(payload, mimetype='application/octet-stream',
                        headers={'X-Num-Points': str(len(pts)), 'Cache-Control': 'max-age=3600'})

    @app.route('/api/sample/<token>/colors')
    def colors(token):
        mode = request.args.get('mode', 'distance')
        cmap = request.args.get('cmap', 'jet')
        modality = request.args.get('modality', 'none')
        if mode not in COLOR_MODES or cmap not in COLORMAPS or modality not in MODALITIES:
            abort(400, 'bad mode / cmap / modality')
        rgb = browser.colors(_sample(token), mode, cmap, modality, _float('max_range', None))
        return Response(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes(),
                        mimetype='application/octet-stream',
                        headers={'X-Num-Points': str(len(rgb)), 'Cache-Control': 'max-age=3600'})

    @app.route('/api/categories')
    def categories():
        return jsonify(browser.categories())

    @app.route('/api/scene/<token>/instances')
    def instances(token):
        try:
            browser.db.get('scene', token)
        except KeyError:
            abort(404)
        return jsonify(browser.instances(token))

    @app.route('/api/sample/<token>/boxes')
    def boxes(token):
        return jsonify(browser.boxes(_sample(token)))

    @app.route('/api/sample/<token>/boxes2d/<channel>')
    def boxes_2d(token, channel):
        data = browser.boxes_2d(_sample(token), channel)
        if data is None:
            abort(404, f'no {channel} in this sample')
        return jsonify(data)

    @app.route('/api/sample/<token>/radar')
    def radar(token):
        return jsonify(browser.radar(_sample(token)))

    @app.route('/api/sample/<token>/image/<channel>')
    def image(token, channel):
        data = browser.image(_sample(token), channel)
        if data is None:
            abort(404, f'no {channel} image for this sample')
        img, K, T, dist, scale = data
        if request.args.get('overlay', '0') == '1':
            pts, _ = browser.cloud(token)
            rgb = browser.colors(token, request.args.get('mode', 'distance'),
                                 request.args.get('cmap', 'jet'), 'none', _float('max_range', None))
            radius = 1 if img.shape[1] < 700 else 2
            img = overlay_points(img, pts, rgb, K, T, dist, scale, radius=radius,
                                 max_range=_float('max_range', None))
        return Response(encode_jpeg(img), mimetype='image/jpeg',
                        headers={'Cache-Control': 'max-age=3600'})

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default='data/useful')
    ap.add_argument('--version', default='v1.1')
    ap.add_argument('--host', default='127.0.0.1', help='bind address (127.0.0.1 + ssh -L is the safe default)')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args(argv)

    browser = SceneBrowser(dataroot=args.dataroot, version=args.version)
    app = create_app(browser)
    print(KEYMAP)
    print(f'USEFUL viewer: {len(browser.scenes)} scenes from {os.path.abspath(args.dataroot)} ({args.version})')
    print(f'  open   http://{args.host}:{args.port}')
    print(f'  remote ssh -N -L {args.port}:localhost:{args.port} <user>@{os.uname().nodename}   '
          f'then http://localhost:{args.port}')
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
