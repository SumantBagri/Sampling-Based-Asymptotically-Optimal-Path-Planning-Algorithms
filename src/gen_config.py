#!/usr/bin/env python3

import glob
import json
import numpy as np

from fmt.utils import load_image

MAPS_DIR = "../Maps/evaluation_data"
CONFIG_DIR = "config"

# Map idex for: alternating gaps, bugtrapforest, forest, gapsforest, maze, multibugtrap, bugtrap
MAP_IDXS = [2, 10, 13, 19, 24, 28, 37]
RUNS_PER_MAP = 5
SEPARATION_FACTORS = [0.5, 0.75]
STATE_PAIR_PER_MAP = len(SEPARATION_FACTORS)
NEIGHBOUR_EPS = 0.1

config = {
    "rng_seeds" : np.random.randint(999, size=RUNS_PER_MAP), # seed for sampling
    "max_num_steps": 2000, # maximum number of exploration steps for a single run before giving up
    "map_idxs": MAP_IDXS, # map indexes to be used for simulations
    "state_pair_per_map": STATE_PAIR_PER_MAP, # number of start-end pairs for each map
    "runs_per_map": RUNS_PER_MAP, # number of runs per map
    "separation_factor": SEPARATION_FACTORS, # minimum distance between start-end states for each map
    "neighbour_eps": NEIGHBOUR_EPS, # epsilon neighbourhood for finding start-end state pair for each map
    "col_dst": 1.0, # min dist from obstacle for collision check
    "fmt": {
        "path_res": 0.1, # collision check for fmt
        "heuristic_weights": [0.0, 1.0], # heuristic w eights for euclidean heuristics
        "batch_size": [5, 10, 20, 40, 80, 100, 300, 600, 1000],
        "start_idx": 10000, # global starting index for fmt*
    },
    "bit": {
        "max_div": 50, # split for doing collision check (bit, nrrt)
        "start_idx": 20000, # global starting index for bit*
    },
    "nrrt": {
        "max_div": 200, # split for doing collision check (bit, nrrt)
        "batch_size": [5, 10, 20, 40, 80, 100, 300, 600, 1000],
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

def ed(s, t):
    return np.linalg.norm(np.subtract(s,t))

def main():
    for i, map in enumerate(sorted(glob.glob(f"{MAPS_DIR}/*"))):
        # Load the world
        world = load_image(map)
        # check if its black and white
        assert (np.unique(world.ravel()) == np.array([0, 255])).all()
        
        # If all good above then proceed to populate the config dict
        config["maps"][f"map{i}"] = {"path": map}
        world = world[:,:,0]
        diag = np.sqrt(np.square(world.shape[0]-1)+np.square(world.shape[1]-1))
        # Get the free world (pixels with value 255)
        free_world = np.argwhere(world == 255)
        run = 0
        while run < STATE_PAIR_PER_MAP:
            s, t = np.random.randint(free_world.shape[0], size=2)
            start = list(free_world[s])
            target = list(free_world[t])
            dist = ed(start, target)
            if is_free(start, world) and is_free(target, world) and dist > SEPARATION_FACTORS[run]*diag - NEIGHBOUR_EPS and dist < SEPARATION_FACTORS[run]*diag + NEIGHBOUR_EPS:
                config["maps"][f"map{i}"][f"s{run}"] = {"start": start, "target": target}
                run += 1

    # Convert config dictionary to json file 
    with open(f"{CONFIG_DIR}/config.json", "w") as f:
        json.dump(config, f, cls=NpEncoder)

if __name__ == "__main__":
    main()