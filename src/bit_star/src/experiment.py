from main import main


if __name__ == "__main__":

    args_template = ['main.py']
    configs_path = "../config/evaluation_data/"

    for i in range(1, 4):
        # Do 5 runs of each

        config_file_path = configs_path + f"map{i}Config.JSON"
        args = args_template.copy()
        args.append(config_file_path)
        args.append("False")

        main(args)
