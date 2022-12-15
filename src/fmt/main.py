#!/usr/bin/env python3

import click
import glob
import json
from tqdm import tqdm

from data import PathPlanningCSVOperator
from FMTPlanner import FMTPlanner
from utils import load_image


@click.group()
def main():
    """
    Runs the FMT* algorithm on given maps (config/stdin) to find a path
    """
    pass


@main.command(help="Test Mode")
@click.option('--hw', default=0.0, help='Weight of the euclidean heuristics')
@click.argument('map_idx')
def test(map_idx, hw):
    """
    Runs FMT* algorithm on test maps

    MAP_IDX: Index of the map to use from ../Maps/test_data/
    """
    rgbworld = load_image(glob.glob(f'../Maps/test_data/*{map_idx}*')[0])
    planner = FMTPlanner(world=rgbworld)
    CSVOperator = PathPlanningCSVOperator('fmt', 'test', f'_hw_{hw}')

    with open('map_states.json') as f:
        states = json.load(f)[f'map{map_idx}']
    for i, s in enumerate(states):
        data_dict = {
            'Overall Test Number': f"{i}",
            'Algorithm': 'fmt',
            'Map Type': 'test',
            'Map Id': f"{map_idx}",
            'Start Point': str(states[s]['start']),
            'Goal Point': str(states[s]['target']),
            'Test Number': f"{i}",
            'Batch Size': "1000",
            'Cumulative Num Sampled': "1000",
        }
        plan_info = planner.plan(start=states[s]['start'],
                                 target=states[s]['target'],
                                 map_idx=map_idx,
                                 pidx=i,
                                 hw=hw)
        data_dict["Iteration"] = plan_info['num_iters']
        data_dict["Timestep"] = plan_info['total_execution_time']
        data_dict["Num Collision Checks"] = plan_info['collision_checks']
        data_dict["Current Path Cost"] = plan_info['cost']
        data_dict["Any Path Found"] = plan_info['path_found']
        data_row = [p[1] for p in sorted(
            data_dict.items(), key=lambda p: CSVOperator.dcols.index(p[0]))]
        CSVOperator.writerow(data_row)
        # print(f"Total Cost: {plan_info['cost']}, Execution Time: {plan_info['total_execution_time']}, Collision Checks: {plan_info['collision_checks']}")


@main.command(help="Evaluation Mode (fixed sample size=1000)")
def evaluation():
    """
    Runs FMT* algorithm on evaluation maps

    Batch size is fixed: 1000
    """
    with open('../config/config.json', 'r') as f:
        cfg = json.load(f)
    for hw in tqdm(cfg["fmt"]["heuristic_weights"], position=0, leave=False):
        CSVOperator = PathPlanningCSVOperator('fmt', 'evaluation', f'_hw_{hw}')
        # Clear data from stale run
        CSVOperator.clear()
        write_idx = cfg["fmt"]["start_idx"]
        for n_samples in tqdm(cfg["fmt"]["batch_size"], position=1, leave=False):
            for map_idx in tqdm(cfg["map_idxs"], position=2, leave=False):
                map_dict = cfg["maps"][f"map{map_idx-1}"]
                p = map_dict["path"]
                splits = p.split(".")[-2].split("_")
                map_idx = splits[-2]
                rgbworld = load_image(p)
                for st_pair_num in tqdm(range(cfg["state_pair_per_map"]), position=3, leave=False):
                    sp = map_dict[f"s{st_pair_num}"]['start']
                    tp = map_dict[f"s{st_pair_num}"]['target']
                    for run_num in tqdm(range(cfg["runs_per_map"]), position=4, leave=False):
                        planner = FMTPlanner(world=rgbworld,
                                             n_samples=n_samples,  # change for sample-based analysis,
                                             max_iter=cfg["max_num_steps"],
                                             col_dst=cfg["col_dst"],
                                             pr=cfg["fmt"]["path_res"],
                                             seed=cfg["rng_seeds"][run_num])
                        data_dict = {
                            'Overall Test Number': f"{write_idx}",
                            'Algorithm': 'fmt',
                            'Map Type': f'{splits[-1]}',
                            'Map Id': f"{map_idx}",
                            'Start Point': str(sp),
                            'Goal Point': str(tp),
                            'Test Number': f"{run_num}",
                            'Batch Size': f"{n_samples}",
                            'Cumulative Num Sampled': f"{n_samples}",
                        }
                        plan_info = planner.plan(start=sp,
                                                 target=tp,
                                                 map_idx=map_idx,
                                                 pidx=run_num,
                                                 sidx=st_pair_num,
                                                 mode="evaluation",
                                                 hw=hw)
                        data_dict["Iteration"] = plan_info['num_iters']
                        data_dict["Timestep"] = plan_info['total_execution_time']
                        data_dict["Num Collision Checks"] = plan_info['collision_checks']
                        data_dict["Current Path Cost"] = plan_info['cost']
                        data_dict["Any Path Found"] = plan_info['path_found']
                        data_row = [p[1] for p in sorted(
                            data_dict.items(), key=lambda p: CSVOperator.dcols.index(p[0]))]
                        CSVOperator.writerow(data_row)
                        write_idx += 1


@main.command(help="Evaluation Mode with varying sample size (as defined in the configuration file)")
def eval_on_sample_size():
    pass


if __name__ == "__main__":
    main()
