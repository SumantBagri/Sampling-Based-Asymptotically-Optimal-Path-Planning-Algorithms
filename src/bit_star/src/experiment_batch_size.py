from main import main
import json


if __name__ == "__main__":

    args_template = ['main.py']
    configs_path = "../config/evaluation_data_batch_size/"
    batch_sizes = [[10, 0, 100], [10, 20, 100], [10, 30, 200], [30, 0, 100], [
        30, 50, 200], [30, 100, 300], [50, 0, 100], [50, 50, 200], [50, 100, 400], [30, 10, 100]]

    i = 26
    i = i-1
    # 5 Combinations of start/goal states
    for j in range(0, 2):
        config_file_path = configs_path + f"map{i}Config{j}.JSON"

        # Arguments to Main
        args = args_template.copy()
        args.append(config_file_path)
        args.append("False")
        args.append("../results/evaluation_data_batch_sizes/images/")

        # For each batch size
        for b_size in batch_sizes:
            # Modify Config File
            with open(config_file_path) as config_file:
                config_data = json.load(config_file)

            config_data['batch_size'] = b_size

            with open(config_file_path, "w") as outfile:
                json.dump(config_data, outfile)

            # 5 Runs of each
            for k in range(0, 5):
                main(args)
