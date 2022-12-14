from main import main
import json
import numpy as np


if __name__ == "__main__":

    args_template = ['main.py']
    configs_path = "../config/evaluation_data/"

    with open("../config/config.json") as main_config_file:
        main_config_data = json.load(main_config_file)

    map_idxs = main_config_data["map_idxs"]
    map_idxs = np.subtract(map_idxs, 1)

    # Maps
    for i in map_idxs:
        # 2 Combinations of start/goal states
        for j in range(0, 2):
            config_file_path = configs_path + f"map{i}Config{j}.JSON"
            args = args_template.copy()
            args.append(config_file_path)
            args.append("False")
            args.append("../results/evaluation_data/images/")
            # 5 Runs of each
            for k in range(0, 5):
                main(args)
