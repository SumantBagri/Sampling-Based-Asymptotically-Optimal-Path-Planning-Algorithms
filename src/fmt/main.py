import glob
import json
import sys

from FMTPlanner import FMTPlanner
from utils import load_image

def main(map_idx, hw=0.0):
    rgbworld = load_image(glob.glob(f'Maps/*{map_idx}*')[0])
    planner = FMTPlanner(world=rgbworld, n_samples=1000)

    f = open('map_states.json')
    states = json.load(f)[f'map{map_idx}']
    for i, s in enumerate(states):
        plan_info = planner.plan(states[s]['start'], states[s]['target'], map_idx, i, hw)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py FILENAME [HEURISTIC_WEIGHT]")
        sys.exit(1)
    
    if len(sys.argv) == 3:
        main(sys.argv[1], float(sys.argv[2]))
    else:
        main(sys.argv[1])