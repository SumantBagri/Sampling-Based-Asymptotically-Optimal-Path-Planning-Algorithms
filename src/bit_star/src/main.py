import sys
import pickle
import cv2

import matplotlib.pyplot as plt
import pandas as pd
import re
import json

from bitstar_planner import BITPlanner
from state import State


def main(args):

    if len(args) < 2:
        print("Usage: main.py config_file.JSON")
        sys.exit(1)

    if not (args[1].lower().endswith('.json')):
        print("Usage: Argument 1 must be a config file of .json type")
        sys.exit(1)

    # Whether or not to show the planning algorithm
    show_img = True
    if len(args) >= 3:
        if args[2].lower() == "false":
            show_img = False

    # Path for output images of the planned path
    imgs_path = "../results/images/"
    if len(args) == 4:
        imgs_path = args[3]

    # Config Data
    with open(args[1]) as config_file:
        config_data = json.load(config_file)

    overall_test_num = config_data['overall_test_num']
    max_num_steps = config_data['max_num_steps']
    # [starting batch size, batch size increment, max batch size]
    batch_size = config_data['batch_size']
    map_file = config_data['map_file']
    output_data_file = config_data['output_data_file']
    final_vals_output_data_file = config_data['final_vals_output_data_file']
    start_state_config = config_data['start_state']
    dest_state_config = config_data['dest_state']
    test_num = config_data['test_num']

    # Load the Map
    if map_file.lower().endswith('.pkl'):
        pkl_file = open(map_file, 'rb')
        # world is a numpy array with dimensions (rows, cols, 3 color channels)
        world = pickle.load(pkl_file)
        pkl_file.close()
    elif map_file.lower().endswith(('.png', '.jpeg', '.jpg')):
        # world is a numpy array with dimensions (rows, cols, 3 color channels)
        world = cv2.imread(map_file)
    else:
        print("Map file type not supported")
        exit()

    bit = BITPlanner(world)

    # Map ID
    re_matches = re.findall('map_(\d+)_(\w+)[.png|.pkl|.jpg|.jpeg]', map_file)
    map_id = re_matches[0][0]
    map_type = re_matches[0][1]

    # Start State
    if (type(start_state_config) is list) and (len(start_state_config) == 2):
        # Try to use given start state
        start_state = State(start_state_config[0], start_state_config[1], None)
        """
        if not (bit.state_is_free(start_state)):
            # Sample
            start_state = bit.sample_state()
        """
    else:
        # Sample
        start_state = bit.sample_state()

    # Destination State
    if (type(dest_state_config) is list) and (len(dest_state_config) == 2):
        # Try to use given dest state
        dest_state = State(dest_state_config[0], dest_state_config[1], None)
        """
        if (not (bit.state_is_free(dest_state))) or bit.path_is_obstacle_free(start_state, dest_state):
            # Sample
            print(
                "The provided Start and Goal States are not useable, sampling new destination state")
            dest_state = bit.sample_state()
            while bit.path_is_obstacle_free(start_state, dest_state):
                dest_state = bit.sample_state()
        """
    else:
        # Sample
        dest_state = bit.sample_state()
        while bit.path_is_obstacle_free(start_state, dest_state):
            dest_state = bit.sample_state()

    # BIT* Algorithm
    plan = bit.plan(start_state,
                    dest_state,
                    max_num_steps,
                    batch_size,
                    overall_test_num,
                    imgs_path,
                    show_img=show_img)

    cv2.destroyAllWindows()

    # Bar Plot of Time Spent on Each Operation
    """
    sorted_time_tracker = sorted(zip(list(bit.time_tracker.values()), list(
        bit.time_tracker.keys())), reverse=True)
    sorted_time_vals = [vals for vals, keys in sorted_time_tracker]
    sorted_time_keys = [keys for vals, keys in sorted_time_tracker]
    plt.bar(range(len(bit.time_tracker)), sorted_time_vals, align='center')
    plt.xticks(range(len(bit.time_tracker)), sorted_time_keys)
    plt.xlabel("Operation")
    plt.ylabel("Time (s)")
    plt.title("Operation vs Time")
    plt.show()
    """

    data = {
        'Overall Test Number': [overall_test_num for iter in range(len(bit.current_time_ex_plotting_elapsed_arr))],
        'Algorithm': ["BIT*" for iter in range(len(bit.current_time_ex_plotting_elapsed_arr))],
        'Map Type': [map_type for iter in range(len(bit.current_time_ex_plotting_elapsed_arr))],
        'Map Id': [map_id for iter in range(len(bit.current_time_ex_plotting_elapsed_arr))],
        'Start Point': [f"({start_state.x}, {start_state.y})" for iter in range(len(bit.current_time_ex_plotting_elapsed_arr))],
        'Goal Point': [f"({dest_state.x}, {dest_state.y})" for iter in range(len(bit.current_time_ex_plotting_elapsed_arr))],
        'Test Number': [test_num for iter in range(len(bit.current_time_ex_plotting_elapsed_arr))],
        'Iteration': [iter for iter in bit.current_iteration_arr],
        'Timestep': bit.current_time_ex_plotting_elapsed_arr,
        'Num Collision Checks': bit.num_collision_checks_arr,
        'Batch Size': [batch_size for iter in range(len(bit.current_time_ex_plotting_elapsed_arr))],
        'Current Batch Size': bit.batch_size_arr,
        'Cumulative Num Sampled': bit.cumulative_sampled_arr,
        'Current Path Cost': bit.current_path_cost_arr,
        'Any Path Found': bit.any_path_found_arr,
    }

    df = pd.DataFrame(data)

    df['Rounded Timestep'] = df['Timestep'].round(1)

    with open(output_data_file, 'a') as f:
        df.to_csv(f, mode='a', index=False, header=f.tell()
                  == 0, line_terminator='\n')

    with open(final_vals_output_data_file, 'a') as f:
        df.iloc[-1:].to_csv(f, mode='a', index=False, header=f.tell()
                            == 0, line_terminator='\n')

    # Overwrite Config Data
    overwritten_config_data = config_data
    overwritten_config_data['overall_test_num'] = int(
        overwritten_config_data['overall_test_num']) + 1
    overwritten_config_data['test_num'] = int(
        overwritten_config_data['test_num']) + 1
    with open(args[1], 'w') as config_file:
        json.dump(overwritten_config_data, config_file, indent=4)


if __name__ == "__main__":

    main(sys.argv)
