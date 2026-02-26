import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import argparse
import numpy as np
import json

from geometry_utils import generate_map
import useful.useful as useful
parser = argparse.ArgumentParser()

parser.add_argument('-p', '--dataroot', help='Path to data')
parser.add_argument('-v', '--version', help='Version of the dataset')
parser.add_argument('-t', '--scene_token', help='Scene token')
parser.add_argument('--marker_first', default=True, help='Set marker for first coordinates in map')
parser.add_argument('--verbose', default=False)

if __name__=='__main__':
    args = parser.parse_args()

    usfl = useful.USEFUL(args.dataroot, args.version, verbose=args.verbose)

    scene_record = usfl.get('scene', args.scene_token)
    log_record = usfl.get('log', scene_record['log_token'])

    current_sample_token = scene_record['first_sample_token']
    coordinates = np.empty((scene_record['nbr_samples'], 2))

    for i in range(scene_record['nbr_samples']):
        sample_record = usfl.get('sample', current_sample_token)
        if i == 0:
            timestamp = sample_record['timestamp']
        coordinates[i,0] = sample_record['ego_pose']['lat_deg']
        coordinates[i,1] = sample_record['ego_pose']['lon_deg']

        current_sample_token = sample_record['next']
    folder = f'{args.dataroot}/maps'
    if not os.path.exists(folder):
        os.mkdir(folder)
    filename= f'{folder}/{log_record["logfile"]}__{timestamp}.html'
    generate_map(coordinates, file_name=filename,start_point=args.marker_first)
    
    if not os.path.exists(usfl.table_root.joinpath(f'map.json')):
        with open(usfl.table_root.joinpath(f'map.json'), "w") as json_file:
            json_file.write("[]")
    
    with open(usfl.table_root.joinpath(f'map.json'), 'r') as map_file:
        map = json.load(map_file)
        map.append({'token': useful.USEFUL.generate_token(),
                    'filename': filename.split('/')[-1],
                    'scene_token': args.scene_token})
        
    with open(usfl.table_root.joinpath(f'map.json'), "w") as json_file:
        json.dump(map, json_file, indent=4)