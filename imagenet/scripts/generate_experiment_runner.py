import os

"""
Use this file to bulk run experiments!

1. Add experiments to the experiment list
2. Change the conda env name in the main function
3. Change the offset value in the main function (to something high) if there are straggling tmux sessions from your previous batch run
4. Run "python scripts/generate_experiment_runner.py"
5. This should generate a scripts/run_experiments.sh, which you can execute for your batch run.
"""
# Specify configs
# self.conv1    |   [224, 224, 3]   |   0
# self.bn1      |   [112, 112, 64]  |   1
# self.relu     |   [112, 112, 64]  |   2
# self.maxpool  |   [56, 56, 64]    |   3
# self.layer1   |   [56, 56, 64]    |   4
# self.layer2   |   [28, 28, 128]   |   5
# self.layer3   |   [14, 14, 256]   |   6
# self.layer4   |   [7, 7, 512]     |   7
# self.avgpool  |   [1, 1, 512]     |   8
# self.flatten  |   [512]           |   9
# self.fc       |   [1000]          |   10
approach_name = '3'
filter_sizes = [3, 5]
strides = [1]
stackings = [1]
injection_points = [
    [6],
    [7],
    [6, 7],
]
positional_encodings = [0, 10, 25]
residuals = ['False','True']
# Aggregate specifications
experiment_list = []
for stacking in stackings:
    for injection_point in injection_points:
        for pos_enc in positional_encodings:
            for residual in residuals:
                for filter_size in filter_sizes:
                    for stride in strides:
                        experiment_list.append(
                            ["resnet", approach_name, filter_size, stride, injection_point, stacking, pos_enc, residual]
                        )


def generate_command(experiment_config, env_name):

    injection_info = [
        [i, experiment_config[5], experiment_config[2]] for i in experiment_config[4]
    ]

    residual_connection_arg = " --use_residual_connection" if experiment_config[7] == 'True' else ""

    out = (
        f"'source /srv/share/gstoica3/miniconda3/etc/profile.d/conda.sh && conda activate {env_name} && "
        "srun -p overcap -A overcap -t 48:00:00"
        + ' --constraint=a40 --gpus-per-node=1 -c 6 python main.py -a "resnet18" /srv/datasets/ImageNet '
        + f' --approach_name "{experiment_config[1]}"'
        + f" --pos_emb_dim {experiment_config[6]}"
        + f' --injection_info "{injection_info}"'
        + f" --stride {experiment_config[3]}"
        + residual_connection_arg
        + "'"
    )

    return out


def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2  # copy R bits to X
    os.chmod(path, mode)


def generate_bash_executable(env_name="p3", offset=0):

    # Set offset value incase tmux sessions have a chance of having duplicate names
    executable = """#!/bin/bash"""

    for i, experiment in enumerate(experiment_list):

        tmux_ind = offset + i

        executable += "\n\n"

        tmux_prefix = f"tmux new-session -d -s ImageNet_{approach_name.replace('_','')}{tmux_ind} "
        executable += tmux_prefix

        command = generate_command(experiment, env_name)
        executable += command

        executable += "\n\n"

        executable += f"echo {command}"

    with open("scripts/run_experiments.sh", "w") as f:
        f.write(executable)

    make_executable("scripts/run_experiments.sh")


if __name__ == "__main__":
    generate_bash_executable(env_name="cifar", offset=400)
