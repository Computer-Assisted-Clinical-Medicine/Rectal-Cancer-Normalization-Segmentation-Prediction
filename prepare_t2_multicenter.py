"""
Prepare the training and run it on the cluster or a local machine (automatically detected)
"""
import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import yaml

# logger has to be set before tensorflow is imported
tf_logger = logging.getLogger("tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# pylint: disable=wrong-import-position, unused-import

from experiment import Experiment
from SegmentationNetworkBasis.architecture import DeepLabv3plus, DenseTiramisu, UNet
from SegmentationNetworkBasis.normalization import NORMALIZING
from SegmentationNetworkBasis.preprocessing import preprocess_dataset
from utils import compare_hyperparameters, export_batch_file, generate_folder_name

# set tf thread mode
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"


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

    # load data
    with open(data_dir / "images.yaml") as f:
        found_images = yaml.load(f, Loader=yaml.Loader)
    with open(data_dir / "segmented_images.yaml") as f:
        segmented_images = yaml.load(f, Loader=yaml.Loader)
    timepoints = pd.read_csv(data_dir / "timepoints.csv", sep=";", index_col=0)

    unsegmented_images = list(
        set(list(found_images.keys())) - set(list(segmented_images.keys()))
    )
    unsegmented_images.sort()

    K_FOLD = 5
    N_CHANNELS = 1

    # ignore certain files
    ignore = ["1005_2_l0_d0"]  # this file has labels not in the image

    # create dict with all points. The names have the format:
    # patient_timepoint_l{label_number}_d{diffusion_number}
    dataset: Dict[str, Dict[str, Union[List[str], str]]] = {}
    for timepoint, data in segmented_images.items():
        for label_num, label_data in enumerate(data):
            assert "T2 axial" in label_data["name"]
            found_data = found_images[timepoint]
            if not "ADC axial recalculated" in found_data:
                continue
            if not "Diff axial b800" in found_data:
                if "Diff axial b800 recalculated" in found_data:
                    B_NAME = "Diff axial b800 recalculated"
                else:
                    raise ValueError("No b800 image found but an ADC image")
            else:
                B_NAME = "Diff axial b800"
            for diff_num, (adc, b800) in enumerate(
                zip(found_data["ADC axial recalculated"], found_data[B_NAME])
            ):
                name = f"{timepoint}_l{label_num}_d{diff_num}"
                if name in ignore:
                    continue
                image = label_data["image"].replace(
                    "Images", "Images registered and N4 corrected"
                )
                dataset[name] = {
                    "images": [data_dir / image],
                    "labels": data_dir / label_data["labels"],
                }

    # add unsegmented before therapy images
    for timepoint in timepoints.query(
        "not segmented & treatment_status == 'before therapy'"
    ).folder:
        found_data = found_images[timepoint]
        if not "T2 axial" in found_data:
            continue
        for num, img in enumerate(found_data["T2 axial"]):
            name = f"{timepoint}_t{num}"
            dataset[name] = {
                "images": [data_dir / image],
            }

    # export dataset
    dataset_file = experiment_dir / "dataset.yaml"
    with open(dataset_file, "w") as f:
        yaml.dump(dataset, f, sort_keys=False)

    # define the parameters that are constant
    train_parameters = {
        "l_r": 0.001,
        "optimizer": "Adam",
        "epochs": 100,
        # parameters for saving the best model
        "best_model_decay": 0.3,
        # scheduling parameters
        "early_stopping": False,
        "patience_es": 15,
        "reduce_lr_on_plateau": False,
        "patience_lr_plat": 10,
        "factor_lr_plat": 0.5,
        # finetuning parameters
        "finetune_epoch": 0,
        "finetune_layers": "all",
        "finetune_lr": 0.001,
        # sampling parameters
        "samples_per_volume": 80,
        "background_label_percentage": 0.15,
        # Augmentation parameters
        "add_noise": False,
        "max_rotation": 0.1,
        "min_resolution_augment": 1.2,
        "max_resolution_augment": 0.9,
    }

    preprocessing_parameters = {
        "resample": True,
        "target_spacing": (1, 1, 3),
    }

    constant_parameters = {
        "train_parameters": train_parameters,
        "preprocessing_parameters": preprocessing_parameters,
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
        "regularize": (True, "L2", 1e-5),
        "drop_out": (True, 0.01),
        "activation": "elu",
        "cross_hair": False,
        "clipping_value": 1,
        "res_connect": True,
        "n_filters": (F_BASE * 8, F_BASE * 16, F_BASE * 32, F_BASE * 64, F_BASE * 128),
        "do_bias": False,
        "do_batch_normalization": True,
        "ratio": 2,
    }
    network_parameters_DenseTiramisu = {
        "regularize": (True, "L2", 1e-5),
        "drop_out": (True, 0.01),
        "activation": "elu",
        "cross_hair": False,
        "clipping_value": 1,
        "layers_per_block": (4, 5, 7, 10, 12),
        "bottleneck_layers": 15,
        "growth_rate": 16,
        "do_bias": False,
        "do_batch_normalization": True,
    }
    network_parameters_DeepLabv3plus = {
        "aspp_rates": (3, 6, 9),  # half because input is half the size
        "clipping_value": 50,
        "backbone": "densenet121",
    }
    network_parameters = [network_parameters_UNet, network_parameters_DeepLabv3plus]
    architectures = [UNet, DeepLabv3plus]

    hyper_parameters_new = []
    for hyp in hyper_parameters:
        for arch, network_params in zip(architectures, network_parameters):
            hyp_new = copy.deepcopy(hyp)
            hyp_new["architecture"] = arch
            hyp_new["network_parameters"] = network_params
            hyper_parameters_new.append(hyp_new)
    hyper_parameters = hyper_parameters_new

    ### normalization method ###
    normalization_methods = {
        NORMALIZING.QUANTILE: {
            "lower_q": 0.05,
            "upper_q": 0.95,
        },
        NORMALIZING.HISTOGRAM_MATCHING: {"mask_quantile": 0},
    }
    # add all methods with their hyperparameters
    hyper_parameters_new = []
    for hyp in hyper_parameters:
        for method, params in normalization_methods.items():
            hyp_new = copy.deepcopy(hyp)
            hyp_new["preprocessing_parameters"]["normalizing_method"] = method
            hyp_new["preprocessing_parameters"]["normalization_parameters"] = params
            hyper_parameters_new.append(hyp_new)
    hyper_parameters = hyper_parameters_new

    ### dimension ###
    dimensions = [2]
    hyper_parameters = vary_hyperparameters(hyper_parameters, ("dimensions",), dimensions)

    ### percent_of_object_samples ###
    pos_values = [0.4]
    hyper_parameters = vary_hyperparameters(
        hyper_parameters, ("train_parameters", "percent_of_object_samples"), pos_values
    )

    attention = [False]
    hyper_parameters = vary_hyperparameters(
        hyper_parameters, ("network_parameters", "attention"), attention
    )

    encoder_attention = [None]
    hyper_parameters = vary_hyperparameters(
        hyper_parameters, ("network_parameters", "encoder_attention"), encoder_attention
    )

    # generate tensorflow command
    tensorboard_command = f'tensorboard --logdir="{experiment_dir.resolve()}"'
    print(f"To see the progress in tensorboard, run:\n{tensorboard_command}")

    # set config
    PREPROCESSED_DIR = Path("data_preprocessed")

    for location in ["Frankfurt", "Regensburg", "Mannheim-not-from-study", "all"]:

        timepoints_mannheim = [
            s for s in segmented_images.keys() if s.startswith("990") and s.endswith("_1")
        ]
        if location in ["Regensburg", "Frankfurt"]:
            # only use before therapy images that are segmented
            timepoints_train = timepoints.query(
                f"treatment_status=='before therapy' & segmented & location=='{location}'"
            ).index
        elif location == "all":
            timepoints_train = (
                list(
                    timepoints.query("treatment_status=='before therapy' & segmented").index
                )
                + timepoints_mannheim
            )
        elif location == "Mannheim-not-from-study":
            timepoints_train = timepoints_mannheim

        experiment_group_name = f"Normalization_{location}"
        current_exp_dir = experiment_dir / experiment_group_name

        # set training files
        train_list = [key for key in dataset if key.partition("_l")[0] in timepoints_train]

        # set test files (just use the rest)
        test_list: List[str] = list(set(dataset.keys()) - set(train_list))

        # set up all experiments
        experiments: List[Experiment] = []
        for hyp in hyper_parameters:
            # use less filters for 3D on local computer
            if not "CLUSTER" in os.environ:
                if hyp["architecture"] is UNet:
                    if hyp["dimensions"] == 3:
                        F_BASE = 4
                        n_filters = (
                            F_BASE * 8,
                            F_BASE * 16,
                            F_BASE * 32,
                            F_BASE * 64,
                            F_BASE * 128,
                        )
                        hyp["network_parameters"]["n_filters"] = n_filters
                    else:
                        F_BASE = 8
                        n_filters = (
                            F_BASE * 8,
                            F_BASE * 16,
                            F_BASE * 32,
                            F_BASE * 64,
                            F_BASE * 128,
                        )
                        hyp["network_parameters"]["n_filters"] = n_filters

            # define experiment (not a constant)
            experiment_name = generate_folder_name(hyp)  # pylint: disable=invalid-name

            # set a name for the preprocessing dir
            # so the same method will also use the same directory
            preprocessing_name = hyp["preprocessing_parameters"]["normalizing_method"].name

            # preprocess data (only do that once for all experiments)
            exp_dataset = preprocess_dataset(
                data_set=dataset,
                num_channels=N_CHANNELS,
                base_dir=experiment_dir,
                preprocessed_dir=PREPROCESSED_DIR / preprocessing_name,
                train_dataset=train_list,
                preprocessing_parameters=hyp["preprocessing_parameters"],
            )

            exp = Experiment(
                hyper_parameters=hyp,
                name=experiment_name,
                output_path_rel=Path(experiment_group_name) / experiment_name,
                data_set=exp_dataset,
                crossvalidation_set=train_list,
                external_test_set=test_list,
                folds=K_FOLD,
                seed=42,
                num_channels=N_CHANNELS,
                folds_dir_rel=Path(experiment_group_name) / "folds",
                tensorboard_images=True,
                versions=["best"],
            )
            experiments.append(exp)

        # export all hyperparameters
        compare_hyperparameters(experiments, current_exp_dir)

        # if on cluster, export slurm files
        if "CLUSTER" in os.environ:
            slurm_files = []
            working_dir = Path("").resolve()
            if not working_dir.exists():
                working_dir.mkdir()
            for exp in experiments:
                slurm_files.append(exp.export_slurm_file(working_dir))

            start_all_batch = current_exp_dir / "start_all_jobs.sh"
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

            # set the environment (might be changed for each machine)
            ps_script_set_env = experiment_dir / "set_env.ps1"
            script_dir = Path(sys.argv[0]).resolve().parent
            COMMAND = f'$env:script_dir="{script_dir}"\n'
            COMMAND += "$env:script_dir=$env:script_dir -replace ' ', '` '\n"
            COMMAND += f'$env:data_dir="{data_dir}"\n'
            COMMAND += f'$env:experiment_dir="{experiment_dir}"\n'

            # create env file
            with open(ps_script_set_env, "w+") as powershell_file_tb:
                powershell_file_tb.write(COMMAND)

            ps_script = current_exp_dir / "start.ps1"
            ps_script_tb = current_exp_dir / "start_tensorboard.ps1"
            ps_script_combine = current_exp_dir / "start_combine.ps1"
            ps_script_analysis = current_exp_dir / "start_analysis.ps1"

            # make a powershell command, add env
            COMMAND = "$script_parent = (get-item $PSScriptRoot ).parent.FullName\n"
            COMMAND += '$set_env="${script_parent}\\set_env.ps1"\n'
            COMMAND += "$set_env=$set_env -replace ' ', '` '\n"
            COMMAND += "Invoke-Expression ${set_env}\n"
            COMMAND += 'Write-Output "Data dir: $env:data_dir"\n'
            COMMAND += 'Write-Output "Experiment dir: $env:experiment_dir"\n'
            COMMAND += 'Write-Output "Script dir: $env:script_dir"\n'

            # activate
            COMMAND += 'Write-Output "Activate Virtual Environment"\n'
            COMMAND += '$activate=${env:script_dir} + "\\venv\\Scripts\\activate.ps1"\n'
            COMMAND += "Invoke-Expression ${activate}\n"

            # tensorboard command (up to here, it is the same)
            COMMAND_TB = COMMAND
            COMMAND_TB += "$start='tensorboard --logdir=\"' + "
            COMMAND_TB += f"${{env:experiment_dir}} + '\\{experiment_group_name}\"'\n"
            COMMAND_TB += "Write-Output $start\n"
            COMMAND_TB += "Invoke-Expression ${start}\n"

            # run combine
            COMMAND_COMBINE = COMMAND
            COMMAND_COMBINE += '$script=${env:script_dir} + "\\combine_models.py"\n'
            COMMAND_COMBINE += (
                f'$output_path=${{env:experiment_dir}} + "\\{experiment_group_name}"\n'
            )
            COMMAND_COMBINE += "$output_path=$output_path -replace ' ', '` '\n"
            COMMAND_COMBINE += '$command="python " + ${script} + " -p ${output_path}"\n'
            COMMAND_COMBINE += "Invoke-Expression ${command}\n"

            # run analysis
            COMMAND_ANALYSIS = COMMAND
            COMMAND_ANALYSIS += '$script=${env:script_dir} + "\\analyze_results.py"\n'
            COMMAND_ANALYSIS += (
                f'$output_path=${{env:experiment_dir}} + "\\{experiment_group_name}"\n'
            )
            COMMAND_ANALYSIS += "$output_path=$output_path -replace ' ', '` '\n"
            COMMAND_ANALYSIS += '$command="python " + ${script} + " -p ${output_path}"\n'
            COMMAND_ANALYSIS += "Invoke-Expression ${command}\n"

            # add the experiments
            COMMAND += '$script=${env:script_dir} + "\\run_single_experiment.py"\n'
            for exp in experiments:
                COMMAND += f'\n\nWrite-Output "starting with {exp.name}"\n'
                exp_p_rel = f"\\{experiment_group_name}\\{exp.output_path.name}"
                COMMAND += f'$output_path=${{env:experiment_dir}} + "{exp_p_rel}"\n'
                for fold_num in range(K_FOLD):
                    COMMAND += f'$command="python " + ${{script}} + " -f {fold_num} -e " + \'${{output_path}}\'\n'
                    COMMAND += "Invoke-Expression ${command}\n"

            with open(ps_script, "w+") as powershell_file:
                powershell_file.write(COMMAND)

            # create tensorboard file
            with open(ps_script_tb, "w+") as powershell_file_tb:
                powershell_file_tb.write(COMMAND_TB)

            # create combine file
            with open(ps_script_combine, "w+") as powershell_file_combine:
                powershell_file_combine.write(COMMAND_COMBINE)

            # create analysis file
            with open(ps_script_analysis, "w+") as powershell_file_analysis:
                powershell_file_analysis.write(COMMAND_ANALYSIS)

            print(f"To run the training, execute {ps_script}")
            print(f"To run tensorboard, execute {ps_script_tb}")
            print(f"To analyse the results, execute {ps_script_analysis}")
