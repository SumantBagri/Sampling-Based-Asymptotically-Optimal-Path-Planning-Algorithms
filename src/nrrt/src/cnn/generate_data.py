#!/usr/bin/env python3
import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
module_dir = os.path.dirname(current_dir) + "/base_planners/"
sys.path.insert(0, module_dir)

import cv2
import numpy as np
from tqdm import tqdm
from astar_planner import AStarPlanner, State


INPUT_DIR = "../map_dataset"
OUTPUT_DIR = "../map_labels"
IMG_SIZE = 256
NUM_RUNS = 200
NUM_IMGS = 50


def get_random_states(img, idx_start=None):    
    '''
        generate NUM_RUNS random states to be
        used as start/end states
    '''
    state_space = np.where(img == 1)
    p = np.random.permutation(len(state_space[0]))
    state_space = (state_space[0][p], state_space[1][p])
    sampling_arr = state_space[0]
    
    # excluding starting states
    if idx_start is not None:
        mask = np.ones(state_space[0].shape, dtype='bool')
        mask[idx_start] = 0
        sampling_arr = state_space[0][mask]

    idx = np.random.choice(sampling_arr, NUM_RUNS, replace=False)
    
    return [State(a, b) for a, b in zip(state_space[0][idx], state_space[1][idx])], idx


if __name__ == "__main__":
     map_types = ["alternating_gaps", "forest", "mazes", "shifting_gaps", "bugtrap_forest", 
                  "gaps_and_forest", "multiple_bugtraps", "single_bugtrap"]

    for map_type in map_types:
        imgs = os.listdir("{}/{}/train".format(INPUT_DIR, map_type))
        imgs = np.random.choice(imgs, size=NUM_IMGS)
        print('processing {}: {} imgs'.format(map_type, len(imgs)))
    
    imgs = os.listdir(INPUT_DIR)
    tq_obj = tqdm(imgs)

    for img_path in tq_obj:
        read current image
        img = cv2.imread("{}/{}/train/{}".format(INPUT_DIR, map_type, img_path), 0)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # 0 - occupied, 1 - free for A-star planner
        img = np.where(img !=255, 0, 1)
        astar = AStarPlanner(img)

        # sample start/end states
        start_states, idx_start = get_random_states(img)
        end_states, _ = get_random_states(img, idx_start)
        prob = np.zeros((IMG_SIZE, IMG_SIZE))
        
        for start, end in zip(start_states, end_states):
            # skip if start/end state not free or no path to dest
            try:
                nodes = astar.plan(start, end)
                nodes_x, nodes_y = zip(*nodes)
            except:
                continue
                
            # assign binary values for astar path
            prob[nodes_x, nodes_y] = 1
            np.save('{}/{}/{}_{}_{}.npy'.format(OUTPUT_DIR, map_type, img_path.split('.')[0], start, end), prob)
            