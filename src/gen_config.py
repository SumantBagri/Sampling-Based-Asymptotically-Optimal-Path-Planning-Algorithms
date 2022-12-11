#!/usr/bin/env python3

import glob
import json
import numpy as np

from utils import load_image

MAPS_DIR = "../Maps/evaluation_data"
CONFIG_DIR = "config"

RUNS_PER_MAP = 5

config = {
    "max_num_steps": 2000, # maximum number of exploration steps for a single run before giving up
    "fmt": {
        "path_res": 0.1, # collision check for fmt
        "col_dst": 1.0, # min dist from obstacle for fmt collision check
        "batch_size": [10, 100, 200, 500, 1000, 2000, 5000, 10000],
    },
    "bit": {
        "max_div": 50, # split for doing collision check (bit, nrrt)
        "col_dst": 1.0, # min dist from obstacle for bit collision check
    },
    "nrrt": {
        "max_div": 200, # split for doing collision check (bit, nrrt)
        "col_dst": 1.0, # min dist from obstacle for nrrt collision check
        "batch_size": [10, 100, 200, 500, 1000, 2000, 5000, 10000],
    },
    "maps": {}
}
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():
    for i, map in enumerate(sorted(glob.glob(f"{MAPS_DIR}/*"))):
        # Load the world
        world = load_image(map)
        # check if its black and white
        assert (np.unique(world.ravel()) == np.array([0, 255])).all()
        
        # If all good above then proceed to populate the config dict
        config["maps"][f"map{i}"] = {"path": map}
        world = world[:,:,0]
        # Get the free world (pixels with value 255)
        free_world = np.argwhere(world == 255)
        for run in range(RUNS_PER_MAP):
            s, t = np.random.randint(free_world.shape[0], size=2)
            config["maps"][f"map{i}"][f"s{run}"] = {"start": list(free_world[s]), "target": list(free_world[t])}

    # Convert config dictionary to json file 
    with open(f"{CONFIG_DIR}/config.json", "w") as f:
        json.dump(config, f, cls=NpEncoder)

if __name__ == "__main__":
    main()