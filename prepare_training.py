"""
Prepare the training and run it on the cluster or a local machine (automatically detected)
"""
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# logger has to be set before tensorflow is imported
tf_logger = logging.getLogger("tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# pylint: disable=wrong-import-position

from experiment import Experiment
from SegmentationNetworkBasis.architecture import UNet, DenseTiramisu
from SegmentationNetworkBasis.segbasisloader import NORMALIZING
from utils import (
    compare_hyperparameters,
    generate_folder_name,
    export_batch_file,
)

# set tf thread mode
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

DEBUG = False
if DEBUG:
    # run everything eagerly
    import tensorflow as tf

    # tf.config.run_functions_eagerly(True)
    # do numeric checks (for NaNs)
    tf.debugging.enable_check_numerics(stack_height_limit=60, path_length_limit=100)


def vary_hyperparameters(parameters, keys: tuple, values: list) -> List[Dict[str, Any]]:
    """Vary the hyperparameter given by the keys by the values given. A list of
    parameters as a dict is taken as input and the returned list will be
    len(values) times bigger than the input list. The hyperparameter will be added
    for each existing entry.

    Parameters
    ----------
    parameters : List[Dict[str, Any]]
        The current hyperparameters
    keys : tuple
        The parameter to vary with the dict keys as tuple, this is then used to
        write to a nested dict, with the first value as key for the first dict.
        All keys besides the last have to already exist
    values : list
        list of values to set as parameter

    Returns
    -------
    List[Dict[str, Any]]
        The new hyperparameter list with increased size
    """
    params_new = []
    for param_set in parameters:
        for val in values:
            param_set_new = copy.deepcopy(param_set)
            to_vary = param_set_new
            for k in keys[:-1]:
                to_vary = to_vary[k]
            to_vary[keys[-1]] = val
            params_new.append(param_set_new)
    return params_new


if __name__ == "__main__":

    data_dir = Path(os.environ["data_dir"])
    experiment_dir = Path(os.environ["experiment_dir"])
    if not experiment_dir.exists():
        experiment_dir.mkdir()

    # load training files
    train_list = np.loadtxt(data_dir / "train_IDs.csv", dtype="str")
    train_list = np.array([str(t) for t in train_list])

    # load test files
    test_list = np.loadtxt(data_dir / "test_IDs.csv", dtype="str")
    test_list = np.array([str(t) for t in test_list])

    K_FOLD = 5

    # get number of channels from database file
    data_dict_file = data_dir / "dataset.json"
    if not data_dict_file.exists():
        raise FileNotFoundError(f"Dataset dict file {data_dict_file} not found.")
    with open(data_dict_file) as f:
        data_dict = json.load(f)
    n_channels = len(data_dict["modality"])

    # define the parameters that are constant
    train_parameters = {
        "l_r": 0.001,
        "optimizer": "Adam",
        "epochs": 100,
        # scheduling parameters
        "early_stopping": True,
        "patience_es": 15,
        "reduce_lr_on_plateau": True,
        "patience_lr_plat": 5,
        "factor_lr_plat": 0.5,
        # sampling parameters
        "samples_per_volume": 80,
        "background_label_percentage": 0.15,
        # Augmentation parameters
        "add_noise": False,
        "max_rotation": 0,
        "min_resolution_augment": 1,
        "max_resolution_augment": 1,
    }

    data_loader_parameters = {"do_resampling": True}

    constant_parameters = {
        "train_parameters": train_parameters,
        "data_loader_parameters": data_loader_parameters,
        "loss": "DICE",
    }
    # define constant parameters
    hyper_parameters: List[Dict[str, Any]] = [
        {
            **constant_parameters,
        }
    ]

    ### architecture ###
    F_BASE = 8
    network_parameters_UNet = {
        "regularize": [True, "L2", 1e-5],
        "drop_out": [True, 0.01],
        "activation": "elu",
        "cross_hair": False,
        "clipping_value": 1,
        "res_connect": True,
        "n_filters": [F_BASE * 8, F_BASE * 16, F_BASE * 32, F_BASE * 64, F_BASE * 128],
        "do_bias": True,
        "do_batch_normalization": False,
    }
    network_parameters_DenseTiramisu = {
        "regularize": [True, "L2", 1e-5],
        "drop_out": [True, 0.01],
        "activation": "elu",
        "cross_hair": False,
        "clipping_value": 1,
        "layers_per_block": (4, 5, 7, 10, 12),
        "bottleneck_layers": 15,
        "growth_rate": 16,
        "do_bias": False,
        "do_batch_normalization": True,
    }
    network_parameters = [network_parameters_UNet, network_parameters_DenseTiramisu]
    architectures = [UNet, DenseTiramisu]

    hyper_parameters_new = []
    for hyp in hyper_parameters:
        for arch, network_params in zip(architectures, network_parameters):
            hyp_new = copy.deepcopy(hyp)
            hyp_new["architecture"] = arch
            hyp_new["network_parameters"] = network_params
            hyper_parameters_new.append(hyp_new)
    hyper_parameters = hyper_parameters_new

    ### normalization method ###
    normalization_methods = [
        NORMALIZING.QUANTILE,  # NORMALIZING.HM_QUANTILE, NORMALIZING.MEAN_STD
    ]
    hyper_parameters = vary_hyperparameters(
        hyper_parameters,
        ("data_loader_parameters", "normalizing_method"),
        normalization_methods,
    )

    ### dimension ###
    dimensions = [2]
    hyper_parameters = vary_hyperparameters(hyper_parameters, ("dimensions",), dimensions)

    ### percent_of_object_samples ###
    pos_values = [0, 0.1, 0.33, 0.4, 0.5, 0.6, 0.8, 1]
    hyper_parameters = vary_hyperparameters(
        hyper_parameters, ("train_parameters", "percent_of_object_samples"), pos_values
    )

    # generate tensorflow command
    tensorboard_command = f'tensorboard --logdir="{experiment_dir.resolve()}"'
    print(f"To see the progress in tensorboard, run:\n{tensorboard_command}")

    # set config
    PREPROCESSED_DIR = "data_preprocessed"

    # set up all experiments
    experiments: List[Experiment] = []
    for hyp in hyper_parameters:
        # use less filters for 3D on local computer
        if not "CLUSTER" in os.environ:
            if hyp["architecture"] is UNet:
                if hyp["dimensions"] == 3:
                    F_BASE = 4
                    n_filters = [
                        F_BASE * 8,
                        F_BASE * 16,
                        F_BASE * 32,
                        F_BASE * 64,
                        F_BASE * 128,
                    ]
                    hyp["network_parameters"]["n_filters"] = n_filters
                else:
                    F_BASE = 8
                    n_filters = [
                        F_BASE * 8,
                        F_BASE * 16,
                        F_BASE * 32,
                        F_BASE * 64,
                        F_BASE * 128,
                    ]
                    hyp["network_parameters"]["n_filters"] = n_filters

        # define experiment
        experiment_name = generate_folder_name(hyp)  # pylint: disable=invalid-name

        exp = Experiment(
            hyper_parameters=hyp,
            name=experiment_name,
            output_path_rel=experiment_name,
            data_set=train_list,
            external_test_set=test_list,
            folds=K_FOLD,
            seed=42,
            num_channels=n_channels,
            folds_dir_rel="folds",
            preprocessed_dir_rel=PREPROCESSED_DIR,
            tensorboard_images=True,
        )
        experiments.append(exp)

    # export all hyperparameters
    compare_hyperparameters(experiments, experiment_dir)

    # if on cluster, export slurm files
    if "CLUSTER" in os.environ:
        slurm_files = []
        working_dir = Path("").resolve()
        if not working_dir.exists():
            working_dir.mkdir()
        for exp in experiments:
            slurm_files.append(exp.export_slurm_file(working_dir))

        start_all_batch = experiment_dir / "start_all_jobs.sh"
        export_batch_file(
            filename=start_all_batch,
            commands=[f"sbatch {f}" for f in slurm_files],
        )

        # and create some needed directories (without their log dirs, jobs don't start)
        plot_dir_slurm = working_dir / "plots" / "slurm"
        if not plot_dir_slurm.exists():
            plot_dir_slurm.mkdir(parents=True)
        combined_dir_slurm = working_dir / "combined_models" / "slurm"
        if not combined_dir_slurm.exists():
            combined_dir_slurm.mkdir(parents=True)
        print(f"To start the training, execute {start_all_batch}")
    # if on local computer, export powershell start file
    else:
        script_dir = Path(sys.argv[0]).resolve().parent
        ps_script = experiment_dir / "start.ps1"
        ps_script_tb = experiment_dir / "start_tensorboard.ps1"
        ps_script_analysis = experiment_dir / "start_analysis.ps1"

        # make a powershell command
        command = f'$script_dir="{script_dir}"\n'
        # add paths
        command += f'$env:data_dir="{data_dir}"\n'
        command += f'$env:experiment_dir="{experiment_dir}"\n'
        command += '$script=${script_dir} + "\\run_single_experiment.py"\n'
        command += 'echo "Data dir: $env:data_dir"\n'
        command += 'echo "Experiment dir: $env:experiment_dir"\n'
        command += 'echo "Script path: $script"\n'

        # activate
        command += 'echo "Activate Virtual Environment"\n'
        command += '$activate=${script_dir} + "\\venv\\Scripts\\activate.ps1"\n'
        command += "Invoke-Expression ${activate}\n"

        # tensorboard command (up to here, it is the same)
        command_tb = command
        command_tb += "$start='tensorboard --logdir=\"' + ${env:experiment_dir} + '\"'\n"
        command_tb += "echo $start\n"
        command_tb += "Invoke-Expression ${start}\n"
        command_tb += 'read-host "Finished, press ENTER to close."'

        # run analysis
        command_analysis = command
        command_analysis += '$script=${script_dir} + "\\analyze_results.py"\n'
        command_analysis += '$command="python " + ${script}\n'
        command_analysis += "Invoke-Expression ${command}\n"
        command_analysis += 'read-host "Finished, press ENTER to close."'

        # add the experiments
        command += '$script=${script_dir} + "\\run_single_experiment.py"\n'
        for exp in experiments:
            command += f'echo "starting with {exp.name}"\n'
            command += (
                f'$output_path=${{env:experiment_dir}} + "\\{exp.output_path.name}"\n'
            )
            for fold_num in range(K_FOLD):
                command += f'$command="python " + ${{script}} + " -f {fold_num} -e " + \'${{output_path}}\'\n'
                command += "Invoke-Expression ${command}\n"
        command += 'read-host "Finished, press ENTER to close."'

        with open(ps_script, "w+") as powershell_file:
            powershell_file.write(command)

        # create tensorboard file
        with open(ps_script_tb, "w+") as powershell_file_tb:
            powershell_file_tb.write(command_tb)

        # create analysis file
        with open(ps_script_analysis, "w+") as powershell_file_analysis:
            powershell_file_analysis.write(command_analysis)

        print(f"To run the training, execute {ps_script}")
        print(f"To run tensorboard, execute {ps_script_tb}")
        print(f"To analyse the results, execute {ps_script_analysis}")