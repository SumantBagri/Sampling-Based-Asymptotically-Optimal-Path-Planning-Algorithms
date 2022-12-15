import json

# Config Data
with open("../config/config.json") as config_file:
    config_data = json.load(config_file)

max_num_steps = config_data['max_num_steps']
start_idx = config_data['bit']['start_idx']
maps_config_data = config_data['maps']

batch_size = [30, 0, 500]
#batch_size = [20, 0, 500]
output_data_file = "../results/evaluation_data/test.csv"
final_vals_output_data_file = "../results/evaluation_data/test_final_vals.csv"
output_config_files_path = "../config/evaluation_data/"


# Data to be written
map_config_data_template = {
    "max_num_steps": max_num_steps,
    "batch_size": batch_size,
    "output_data_file": output_data_file,
    "final_vals_output_data_file": final_vals_output_data_file
}

current_idx = 0

for i in range(0, 40):
    #
    map_config_data = maps_config_data[f'map{i}']
    map_config_data_out = map_config_data_template.copy()
    map_config_data_out['map_file'] = map_config_data['path']
    for j in range(0, 2):
        map_config_data_out['start_state'] = [
            map_config_data[f's{j}']['start'][1], map_config_data[f's{j}']['start'][0]]
        map_config_data_out['dest_state'] = [
            map_config_data[f's{j}']['target'][1], map_config_data[f's{j}']['target'][0]]
        map_config_data_out['overall_test_num'] = start_idx + current_idx
        map_config_data_out['test_num'] = 1
        current_idx += 50
        output_config_file_path = output_config_files_path + \
            f"map{i}Config{j}.JSON"
        with open(output_config_file_path, "w") as outfile:
            json.dump(map_config_data_out, outfile)
