#!/usr/bin/env python3

import glob
import json
import numpy as np

from fmt.utils import load_image

MAPS_DIR = "../Maps/evaluation_data"
CONFIG_DIR = "config"

RUNS_PER_MAP = 5

config = {
    "max_num_steps": 2000, # maximum number of exploration steps for a single run before giving up
    "runs_per_map": RUNS_PER_MAP, # number of runs for each map
    "col_dst": 1.0, # min dist from obstacle for collision check
    "fmt": {
        "path_res": 0.1, # collision check for fmt
        "heuristic_weights": [0.0, 1.0], # heuristic w eights for euclidean heuristics
        "batch_size": [10, 100, 200, 500, 1000, 2000, 5000, 10000],
        "start_idx": 10000, # global starting index for fmt*
    },
    "bit": {
        "max_div": 50, # split for doing collision check (bit, nrrt)
        "start_idx": 20000, # global starting index for bit*
    },
    "nrrt": {
        "max_div": 200, # split for doing collision check (bit, nrrt)
        "batch_size": [10, 100, 200, 500, 1000, 2000, 5000, 10000],
        "start_idx": 30000, # global starting index for nrrt*
        "steering_radius": 10 # steering radius for new node search
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

def is_free(p, w):
    pl = p[0] - int(4*config["col_dst"])
    pr = p[0] + int(4*config["col_dst"])
    pu = p[1] - int(4*config["col_dst"])
    pd = p[1] + int(4*config["col_dst"])

    c1 = pl > 0 and w[pl,p[1]] != 0
    c2 = pr < w.shape[1]-1 and w[pr,p[1]] !=0
    c3 = pu > 0 and w[p[0], pu] != 0
    c4 = pd < w.shape[0]-1 and w[p[0], pd] != 0

    return c1 and c2 and c3 and c4

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
        run = 0
        while run < RUNS_PER_MAP:
            s, t = np.random.randint(free_world.shape[0], size=2)
            start = list(free_world[s])
            target = list(free_world[t])
            if is_free(start, world) and is_free(target, world):
                config["maps"][f"map{i}"][f"s{run}"] = {"start": start, "target": target}
                run += 1

    # Convert config dictionary to json file 
    with open(f"{CONFIG_DIR}/config.json", "w") as f:
        json.dump(config, f, cls=NpEncoder)

if __name__ == "__main__":
    main()