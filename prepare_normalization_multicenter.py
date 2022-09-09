"""
Prepare the training and run it on the cluster or a local machine (automatically detected)
"""
import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import yaml

# logger has to be set before tensorflow is imported
tf_logger = logging.getLogger("tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# pylint: disable=wrong-import-position, unused-import

from gan_normalization import GAN_NORMALIZING, train_gan_normalization
from SegClassRegBasis.architecture import DeepLabv3plus, UNet
from SegClassRegBasis.experiment import Experiment
from SegClassRegBasis.normalization import NORMALIZING
from SegClassRegBasis.preprocessing import preprocess_dataset
from SegClassRegBasis.utils import export_experiments_run_files
from utils import generate_folder_name, split_into_modalities

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
    GROUP_BASE_NAME = "Normalization_Experiment"
    exp_group_base_dir = experiment_dir / GROUP_BASE_NAME
    if not exp_group_base_dir.exists():
        exp_group_base_dir.mkdir(parents=True)

    # load data
    with open(data_dir / "images.yaml", encoding="utf8") as f:
        found_images = yaml.load(f, Loader=yaml.Loader)
    with open(data_dir / "segmented_images.yaml", encoding="utf8") as f:
        segmented_images = yaml.load(f, Loader=yaml.Loader)
    timepoints = pd.read_csv(data_dir / "timepoints.csv", sep=";", index_col=0)

    unsegmented_images = list(
        set(list(found_images.keys())) - set(list(segmented_images.keys()))
    )
    unsegmented_images.sort()

    K_FOLD = 5
    N_CHANNELS = 3
    MODALITIES = ("T2 axial", "Diff axial b800", "ADC axial original")

    # ignore certain files
    ignore = ["1005_2_l0_d0"]  # this file has labels not in the image

    # create dict with all points. The names have the format:
    # patient_timepoint_l{label_number}_d{diffusion_number}
    dataset: Dict[str, Dict[str, Union[List[str], str]]] = {}
    for timepoint, data in segmented_images.items():
        for label_num, label_data in enumerate(data):
            assert MODALITIES[0] in label_data["name"]
            found_data = found_images[timepoint]
            if not "ADC axial original" in found_data:
                continue
            if not "Diff axial b800" in found_data:
                if "Diff axial b800 recalculated" in found_data:
                    B_NAME = "Diff axial b800 recalculated"
                else:
                    print("No b800 image found but an ADC image")
                    continue
            else:
                B_NAME = "Diff axial b800"
            for diff_num, (adc, b800) in enumerate(
                zip(found_data["ADC axial original"], found_data[B_NAME])
            ):
                name = f"{timepoint}_l{label_num}_d{diff_num}"
                if name in ignore:
                    continue
                image = label_data["image"].replace(
                    "Images", "Images registered and N4 corrected"
                )
                b800 = b800.replace("Images", "Images registered and N4 corrected")
                adc = adc.replace("Images", "Images registered and N4 corrected")
                label = label_data["labels"].replace(
                    "Images", "Images registered and N4 corrected"
                )
                dataset[name] = {
                    "images": [image, b800, adc],
                    "labels": label,
                }

    # export dataset
    dataset_file = exp_group_base_dir / "dataset.yaml"
    with open(dataset_file, "w", encoding="utf8") as f:
        yaml.dump(dataset, f, sort_keys=False)

    # define the parameters that are constant
    train_parameters = {
        "l_r": 0.001,
        "optimizer": "Adam",
        "epochs": 100,
        "batch_size": 128,
        "in_plane_dimension": 128,
        # parameters for saving the best model
        "best_model_decay": 0.3,
        # scheduling parameters
        "early_stopping": False,
        "patience_es": 15,
        "reduce_lr_on_plateau": False,
        "patience_lr_plat": 10,
        "factor_lr_plat": 0.5,
        # finetuning parameters
        "finetune_epoch": None,
        # sampling parameters
        "samples_per_volume": 32,
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
        "clip_value": 1,
        "res_connect": True,
        "n_filters": (F_BASE * 8, F_BASE * 16, F_BASE * 32, F_BASE * 64, F_BASE * 128),
        "do_bias": True,
        "do_batch_normalization": False,
        "ratio": 2,
    }
    network_parameters_DeepLabv3plus = {
        "aspp_rates": (3, 6, 9),  # half because input is half the size
        "clip_value": 50,
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
        GAN_NORMALIZING.GAN_DISCRIMINATORS: {
            "depth": 3,
            "filter_base": 16,
            "min_max": False,
        },
        NORMALIZING.HISTOGRAM_MATCHING: {"mask_quantile": 0},
        NORMALIZING.MEAN_STD: {},
        NORMALIZING.HM_QUANTILE: {},
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

    # set config
    PREPROCESSED_DIR = Path(GROUP_BASE_NAME) / "data_preprocessed"

    # set up all experiments
    experiments: List[Experiment] = []

    QUANT_DATASET = None

    for location in [
        "all",
        "Frankfurt",
        "Regensburg",
        "Mannheim",
        "Not-Frankfurt",
        "Not-Regensburg",
        "Not-Mannheim",
    ]:

        if location == "all":
            timepoints_train = list(
                timepoints.query("treatment_status=='before therapy' & segmented").index
            )
            timepoints_train_norm = list(timepoints.index)
        elif location in timepoints.location.unique():
            # only use before therapy images that are segmented
            timepoints_train = timepoints.query(
                f"treatment_status=='before therapy' & segmented & '{location}' in location"
            ).index
            timepoints_train_norm = timepoints.query(f"'{location}' in location").index
        elif "not" in location.lower():
            if location == "Not-Mannheim":
                timepoints_train = timepoints.query(
                    "treatment_status=='before therapy' & segmented & 'Mannheim' not in location"
                ).index
                timepoints_train_norm = timepoints.query("'Mannheim' not in location").index
            else:
                timepoints_train = timepoints.query(
                    f"treatment_status=='before therapy' & segmented & location !='{location.replace('Not-', '')}'"
                ).index
                timepoints_train_norm = timepoints.query(
                    f"location !='{location.replace('Not-', '')}'"
                ).index

        experiment_group_name = f"Normalization_{location}"
        current_exp_dir = experiment_dir / experiment_group_name
        group_dir_rel = Path(GROUP_BASE_NAME) / experiment_group_name

        # set training files
        train_list = [key for key in dataset if key.partition("_l")[0] in timepoints_train]

        # set test files (just use the rest)
        test_list: List[str] = list(set(dataset.keys()) - set(train_list))

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

            # set number of validation files
            hyp["train_parameters"]["number_of_vald"] = max(len(train_list) // 15, 4)

            # define experiment (not a constant)
            experiment_name = generate_folder_name(hyp)  # pylint: disable=invalid-name

            # set a name for the preprocessing dir
            # so the same method will also use the same directory
            pre_params = hyp["preprocessing_parameters"]
            norm_type = pre_params["normalizing_method"]
            preprocessing_name = norm_type.name

            if norm_type == GAN_NORMALIZING.GAN_DISCRIMINATORS:
                PASS_MODALITY = True
                model_paths = []
                for mod_num in range(N_CHANNELS):
                    model_paths.append(
                        train_gan_normalization(
                            timepoints_train=timepoints_train_norm,
                            mod_num=mod_num,
                            preprocessed_dir=PREPROCESSED_DIR,
                            experiment_group=group_dir_rel,
                            modality=MODALITIES[mod_num],
                            **pre_params["normalization_parameters"],
                        )
                    )
                pre_params["normalization_parameters"]["model_paths"] = tuple(model_paths)
                # overlap has already been cut
                CUT_TO_OVERLAP = False
                assert QUANT_DATASET is not None
                # the quant dataset lives in the experiment dir
                DATASET_TO_PROCESS = QUANT_DATASET
                preprocess_base_dir = experiment_dir
            else:
                PASS_MODALITY = False
                CUT_TO_OVERLAP = True
                DATASET_TO_PROCESS = dataset
                preprocess_base_dir = data_dir

            # normalizations that are trained should be saved in the exp group dir
            if norm_type in [
                GAN_NORMALIZING.GAN_DISCRIMINATORS,
                NORMALIZING.HISTOGRAM_MATCHING,
                NORMALIZING.HM_QUANTILE,
            ]:
                preprocessed_dir_exp = (
                    group_dir_rel / "data_preprocessed" / preprocessing_name
                )
            else:
                preprocessed_dir_exp = PREPROCESSED_DIR / preprocessing_name

            # preprocess data (only do that once for all experiments)
            exp_dataset = preprocess_dataset(
                data_set=DATASET_TO_PROCESS,
                num_channels=N_CHANNELS,
                base_dir=experiment_dir,
                data_dir=preprocess_base_dir,
                preprocessed_dir=preprocessed_dir_exp,
                train_dataset=train_list,
                preprocessing_parameters=pre_params,
                pass_modality=PASS_MODALITY,
                cut_to_overlap=CUT_TO_OVERLAP,
            )

            if QUANT_DATASET is None and norm_type == NORMALIZING.QUANTILE:
                QUANT_DATASET = split_into_modalities(exp_dataset, n_channels=N_CHANNELS)

            exp = Experiment(
                hyper_parameters=hyp,
                name=experiment_name,
                output_path_rel=group_dir_rel / experiment_name,
                data_set=exp_dataset,
                crossvalidation_set=train_list,
                external_test_set=test_list,
                folds=K_FOLD,
                seed=42,
                num_channels=N_CHANNELS,
                folds_dir_rel=group_dir_rel / "folds",
                tensorboard_images=True,
            )
            experiments.append(exp)

    # bring experiments in a custom order
    def priority(exp_sort):
        """Define a priority for each experiment"""
        norm = exp_sort.hyper_parameters["preprocessing_parameters"]["normalizing_method"]
        norm_priority = {
            NORMALIZING.QUANTILE: 4,
            GAN_NORMALIZING.GAN_DISCRIMINATORS: 3,
            NORMALIZING.HISTOGRAM_MATCHING: 2,
            NORMALIZING.MEAN_STD: 1,
            NORMALIZING.HM_QUANTILE: 0,
        }

        arch_priority = {
            UNet: 10,
            DeepLabv3plus: 0,
        }
        architecture = exp_sort.hyper_parameters["architecture"]
        return norm_priority[norm] + arch_priority[architecture]

    experiments.sort(key=priority, reverse=True)
    # export all hyperparameters
    export_experiments_run_files(exp_group_base_dir, experiments)
