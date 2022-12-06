"""
Prepare the training and run it on the cluster or a local machine (automatically detected)
"""
import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

# logger has to be set before tensorflow is imported
tf_logger = logging.getLogger("tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from gan_normalization import GAN_NORMALIZING, get_norm_params, train_gan_normalization
from networks import ResNet
from SegClassRegBasis.architecture import DeepLabv3plus, UNet
from SegClassRegBasis.experiment import Experiment
from SegClassRegBasis.normalization import NORMALIZING
from SegClassRegBasis.preprocessing import preprocess_dataset
from SegClassRegBasis.utils import export_experiments_run_files, get_gpu
from utils import (
    create_dataset,
    generate_folder_name,
    get_gan_suffix,
    split_into_modalities,
)

# set tf thread mode
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"


def priority(hp_params, train_loc):
    """Define a priority for each experiment"""
    prio = 0
    arch_priority = {UNet: 2000, DeepLabv3plus: 1000, ResNet: 0}
    architecture = hp_params["architecture"]
    if hp_params["dimensions"] == 2:
        prio += 1000
    prio += arch_priority.get(architecture, 0)

    norm = hp_params["preprocessing_parameters"]["normalizing_method"]
    norm_priority = {
        NORMALIZING.QUANTILE: 400,
        NORMALIZING.HM_QUANTILE: 300,
        GAN_NORMALIZING.GAN_DISCRIMINATORS: 200,
        NORMALIZING.HISTOGRAM_MATCHING: 100,
        NORMALIZING.MEAN_STD: 0,
    }
    prio += norm_priority.get(norm, 0)

    if norm == GAN_NORMALIZING.GAN_DISCRIMINATORS:
        suffix = get_gan_suffix(hp_params)
        suffix_priority = {
            "_3_64_0.50_BetterConv_0.00001_WINDOW_seg": 90,
            "_3_64_0.50_BetterConv_0.00001": 80,
            "_3_64_0.50_BetterConv_0.00001_seg": 70,
            "_3_64_0.50": 60,
        }
        prio += suffix_priority.get(suffix, 0)

    loc_priority = {
        "Frankfurt": 9,
        "Regensburg": 8,
        "Mannheim": 7,
        "all": 6,
        "Not-Frankfurt": 5,
        "Not-Regensburg": 4,
        "Not-Mannheim": 3,
    }
    prio += loc_priority.get(train_loc, 0)

    return prio


if __name__ == "__main__":

    data_dir = Path(os.environ["data_dir"])
    experiment_dir = Path(os.environ["experiment_dir"])
    GROUP_BASE_NAME = "Normalization_Experiment"
    exp_group_base_dir = experiment_dir / GROUP_BASE_NAME
    if not exp_group_base_dir.exists():
        exp_group_base_dir.mkdir(parents=True)

    gpu = tf.device(get_gpu(memory_limit=2000))

    # load data
    timepoints = pd.read_csv(data_dir / "timepoints.csv", sep=";", index_col=0)

    K_FOLD = 5
    N_CHANNELS = 3
    MODALITIES = ("T2 axial", "Diff axial b800", "ADC axial original")

    # ignore certain files
    ignore = ["1005_2_l0_d0"]  # this file has labels not in the image

    # create dict with all points. The names have the format:
    # patient_timepoint_l{label_number}_d{diffusion_number}
    dataset_seg = create_dataset(ignore=ignore)
    dataset_class_reg = create_dataset(ignore=ignore, load_labels=False)

    # export datasets
    dataset_file = exp_group_base_dir / "dataset.yaml"
    with open(dataset_file, "w", encoding="utf8") as f:
        yaml.dump(dataset_seg, f, sort_keys=False)
    dataset_class_reg_file = exp_group_base_dir / "dataset_class_reg.yaml"
    with open(dataset_class_reg_file, "w", encoding="utf8") as f:
        yaml.dump(dataset_class_reg, f, sort_keys=False)

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
        "percent_of_object_samples": 0.4,
        "background_label_percentage": 0.15,
        # Augmentation parameters
        "add_noise": False,
        "max_rotation": 0.1,
        "min_resolution_augment": 1.2,
        "max_resolution_augment": 0.9,
        # no tensorboard callback (slow)
        "write_tensorboard": False,
        # do not save complete models (needs too much storage)
        "save_mode": "weights",
    }

    preprocessing_parameters = {
        "resample": True,
        "target_spacing": (1, 1, 3),
    }

    constant_parameters = {
        "train_parameters": train_parameters,
        "preprocessing_parameters": preprocessing_parameters,
        "loss": {"segmentation": "DICE", "classification": "CEL", "regression": "MSE"},
        "dimensions": 2,
    }
    # define constant parameters
    hyper_parameters: List[Dict[str, Any]] = [
        {
            **constant_parameters,
        }
    ]

    ### architecture ###
    F_BASE = 8
    np_UNet = {
        "regularize": (True, "L2", 1e-5),
        "drop_out": (True, 0.2),
        "activation": "elu",
        "cross_hair": False,
        "clip_value": 1,
        "res_connect": True,
        "n_filters": (F_BASE * 8, F_BASE * 16, F_BASE * 32, F_BASE * 64, F_BASE * 128),
        "do_bias": False,
        "do_batch_normalization": True,
        "ratio": 2,
        "attention": False,
        "encoder_attention": None,
    }
    np_UNet_nb = copy.deepcopy(np_UNet)
    np_UNet_nb["do_batch_normalization"] = False

    np_ResNet = {"weights": None, "resnet_type": "ResNet50"}

    network_parameters = [np_ResNet, np_UNet, np_UNet_nb]
    architectures = [ResNet, UNet, UNet]

    hyper_parameters_new = []
    for hyp in hyper_parameters:
        for arch, network_params in zip(architectures, network_parameters):
            hyp_new = copy.deepcopy(hyp)
            hyp_new["architecture"] = arch
            hyp_new["network_parameters"] = network_params
            if arch is ResNet:
                hyp_new["train_parameters"]["l_r"] = ("exponential", 1e-4, 1e-6)
                hyp_new["network_parameters"]["eval_center"] = True
                hyp_new["train_parameters"]["percent_of_object_samples"] = 0
            hyper_parameters_new.append(hyp_new)
    hyper_parameters = hyper_parameters_new

    hyp_new = {
        **constant_parameters,
    }
    hyp_new["architecture"] = ResNet
    hyp_new["network_parameters"] = np_ResNet.copy()
    hyp_new["network_parameters"]["eval_center"] = True
    hyp_new["train_parameters"]["percent_of_object_samples"] = 0
    hyp_new["train_parameters"]["batch_size"] = 16
    hyp_new["train_parameters"]["in_plane_dimension"] = 64
    hyp_new["train_parameters"]["samples_per_volume"] = 1
    hyp_new["train_parameters"]["number_slices"] = 16
    hyp_new["train_parameters"]["l_r"] = ("exponential", 1e-4, 1e-6)
    hyp_new["dimensions"] = 3
    hyper_parameters.append(hyp_new)

    ### normalization method ###
    normalization_methods = [
        (
            NORMALIZING.QUANTILE,
            {
                "lower_q": 0.05,
                "upper_q": 0.95,
            },
        ),
        (NORMALIZING.HISTOGRAM_MATCHING, {"mask_quantile": 0}),
        (NORMALIZING.MEAN_STD, {}),
        (NORMALIZING.HM_QUANTILE, {}),
        (NORMALIZING.WINDOW, get_norm_params(NORMALIZING.WINDOW)),
        (
            GAN_NORMALIZING.GAN_DISCRIMINATORS,
            {
                "depth": 3,
                "filter_base": 64,
                "min_max": False,
                "smoothing_sigma": 0.5,
                "latent_weight": 1,
                "image_weight": 1,
                "image_gen_weight": 0.5,
                "skip_edges": True,
                "latent": True,
                "train_on_gen": True,
                "disc_type": "BetterConv",
                "batch_size": 128,
                "disc_start_lr": 1e-5,
                "disc_end_lr": 1e-5,
            },
        ),
        (
            GAN_NORMALIZING.GAN_DISCRIMINATORS,
            {
                "depth": 3,
                "filter_base": 64,
                "min_max": False,
                "smoothing_sigma": 0.5,
                "latent_weight": 1,
                "image_weight": 1,
                "image_gen_weight": 0.5,
                "skip_edges": True,
                "latent": True,
                "train_on_gen": True,
                "disc_type": "BetterConv",
                "batch_size": 128,
                "disc_start_lr": 1e-5,
                "disc_end_lr": 1e-5,
                "init_norm_method": NORMALIZING.WINDOW,
            },
        ),
        (
            GAN_NORMALIZING.GAN_DISCRIMINATORS,
            {
                "depth": 3,
                "filter_base": 64,
                "min_max": False,
                "latent_weight": 1,
                "image_weight": 1,
                "image_gen_weight": 0.5,
                "skip_edges": False,
                "latent": True,
                "train_on_gen": True,
                "disc_type": "BetterConv",
                "batch_size": 128,
                "disc_start_lr": 1e-5,
                "disc_end_lr": 1e-5,
                "init_norm_method": NORMALIZING.WINDOW,
            },
        ),
        (
            GAN_NORMALIZING.GAN_DISCRIMINATORS,
            {
                "depth": 3,
                "filter_base": 64,
                "min_max": False,
                "smoothing_sigma": 0.5,
                "latent_weight": 1,
                "image_weight": 1,
                "image_gen_weight": 0.5,
                "skip_edges": True,
                "latent": True,
                "train_on_gen": True,
                "disc_type": "BetterConv",
                "batch_size": 128,
                "disc_start_lr": 1e-5,
                "disc_end_lr": 1e-5,
                "init_norm_method": NORMALIZING.WINDOW,
                "train_on_segmentation": True,
                "unet_parameters": np_UNet,
            },
        ),
        (
            GAN_NORMALIZING.GAN_DISCRIMINATORS,
            {
                "depth": 3,
                "filter_base": 64,
                "min_max": False,
                "smoothing_sigma": 0.5,
                "latent_weight": 1,
                "image_weight": 1,
                "image_gen_weight": 0.5,
                "skip_edges": True,
                "latent": True,
                "train_on_gen": True,
                "disc_type": "BetterConv",
                "batch_size": 128,
                "disc_start_lr": 1e-5,
                "disc_end_lr": 1e-5,
                "train_on_segmentation": True,
                "unet_parameters": np_UNet,
            },
        ),
        (
            GAN_NORMALIZING.GAN_DISCRIMINATORS,
            {
                "depth": 3,
                "filter_base": 64,
                "min_max": False,
                "smoothing_sigma": 0.5,
                "latent_weight": 1,
                "image_weight": 1,
                "image_gen_weight": 0.5,
                "skip_edges": True,
                "latent": True,
                "train_on_gen": True,
                "disc_type": "BetterConv",
                "batch_size": 128,
                "disc_start_lr": 1e-5,
                "disc_end_lr": 1e-5,
                "all_image": True,
            },
        ),
        (
            GAN_NORMALIZING.GAN_DISCRIMINATORS,
            {
                "depth": 3,
                "filter_base": 64,
                "min_max": False,
                "smoothing_sigma": 0.5,
                "latent_weight": 1,
                "image_weight": 1,
                "image_gen_weight": 0.5,
                "skip_edges": True,
                "latent": True,
                "train_on_gen": True,
                "disc_type": "BetterConv",
                "batch_size": 128,
            },
        ),
        (
            GAN_NORMALIZING.GAN_DISCRIMINATORS,
            {
                "depth": 3,
                "filter_base": 64,
                "min_max": False,
                "smoothing_sigma": 0.5,
                "latent_weight": 1,
                "image_weight": 1,
                "image_gen_weight": 0.5,
                "skip_edges": True,
                "latent": True,
                "train_on_gen": True,
            },
        ),
        (
            GAN_NORMALIZING.GAN_DISCRIMINATORS,
            {
                "depth": 3,
                "filter_base": 16,
                "min_max": False,
                "smoothing_sigma": 1,
                "latent_weight": 1,
                "image_weight": 1,
                "image_gen_weight": 0.5,
                "skip_edges": True,
                "latent": True,
                "train_on_gen": True,
            },
        ),
    ]
    # add all methods with their hyperparameters
    hyper_parameters_new = []
    for hyp in hyper_parameters:
        for method, params in normalization_methods:
            hyp_new = copy.deepcopy(hyp)
            hyp_new["preprocessing_parameters"]["normalizing_method"] = method
            hyp_new["preprocessing_parameters"]["normalization_parameters"] = params
            hyper_parameters_new.append(hyp_new)
    hyper_parameters = hyper_parameters_new

    # set config
    PREPROCESSED_DIR = Path(GROUP_BASE_NAME) / "data_preprocessed"

    # set up all experiments
    experiments: List[Experiment] = []
    additional_info: List[Dict[str, Any]] = []

    for location in [
        "Frankfurt",
        "Regensburg",
        "Mannheim",
        "all",
        "Not-Frankfurt",
        "Not-Regensburg",
        "Not-Mannheim",
    ]:

        print(f"Starting with {location}.")

        for hyp in hyper_parameters:

            if hyp["architecture"] is ResNet:
                # use before and after therapy images
                timepoints_to_use = timepoints.query(
                    "treatment_status in ('before therapy', 'before OP')"
                )
            else:
                # only use before therapy images that are segmented
                timepoints_to_use = timepoints.query(
                    "treatment_status=='before therapy' & segmented"
                )

            if location == "all":
                query = "index == index"  # pylint:disable=invalid-name
                N_EPOCHS = 100
            elif location in timepoints_to_use.location.unique():
                if location == "Mannheim":
                    query = "location in ('Mannheim', 'Mannheim-not-from-study')"  # pylint:disable=invalid-name
                else:
                    query = f"'{location}' in location"
                N_EPOCHS = 200
            elif "Not" in location:
                if location == "Not-Mannheim":
                    query = "location not in ('Mannheim', 'Mannheim-not-from-study')"  # pylint:disable=invalid-name
                else:
                    query = f"'{location.partition('-')[2]}' != location"
                N_EPOCHS = 100

            timepoints_train = timepoints_to_use.query(query).index
            timepoints_train_norm = timepoints.query(query).index

            experiment_group_name = f"Normalization_{location}"
            current_exp_dir = experiment_dir / experiment_group_name
            group_dir_rel = Path(GROUP_BASE_NAME) / experiment_group_name

            if hyp["architecture"] is ResNet:

                hyp_dataset = dataset_class_reg

                # set training files
                train_list = [
                    key for key in hyp_dataset if key.partition("_l")[0] in timepoints_train
                ]

                # set test files (just use the rest)
                test_list: List[str] = list(set(hyp_dataset.keys()) - set(train_list))

            else:
                hyp_dataset = dataset_seg

                # set training files
                train_list = [
                    key for key in hyp_dataset if key.partition("_l")[0] in timepoints_train
                ]

                # set test files (just use the rest)
                test_list = list(set(hyp_dataset.keys()) - set(train_list))

            # set number of validation files
            hyp["train_parameters"]["number_of_vald"] = max(len(train_list) // 15, 4)

            # define experiment (not a constant)
            experiment_name = generate_folder_name(hyp)  # pylint: disable=invalid-name

            # set a name for the preprocessing dir
            # so the same method will also use the same directory
            pre_params = hyp["preprocessing_parameters"]
            norm_type = pre_params["normalizing_method"]

            if norm_type == GAN_NORMALIZING.GAN_DISCRIMINATORS:
                # make the base dat set
                pre_params_gan_base = copy.deepcopy(preprocessing_parameters)
                gan_base_norm = pre_params["normalization_parameters"].get(
                    "init_norm_method", NORMALIZING.QUANTILE
                )
                gan_base_params = get_norm_params(gan_base_norm)
                pre_params_gan_base["normalizing_method"] = gan_base_norm
                pre_params_gan_base["normalization_parameters"] = gan_base_params
                DATASET_TO_PROCESS = split_into_modalities(
                    preprocess_dataset(
                        data_set=hyp_dataset,
                        num_channels=N_CHANNELS,
                        base_dir=experiment_dir,
                        data_dir=data_dir,
                        preprocessed_dir=PREPROCESSED_DIR / gan_base_norm.name,
                        train_dataset=train_list,
                        preprocessing_parameters=pre_params_gan_base,
                        pass_modality=False,
                        cut_to_overlap=True,
                    ),
                    n_channels=N_CHANNELS,
                )
                # do the actual normalization
                gan_suffix = get_gan_suffix(hyp)
                preprocessing_name = norm_type.name + gan_suffix
                PASS_MODALITY = True
                model_paths = []
                for mod_num in range(N_CHANNELS):
                    with gpu:
                        model_paths.append(
                            train_gan_normalization(
                                timepoints_train=timepoints_train_norm,
                                mod_num=mod_num,
                                preprocessed_dir=PREPROCESSED_DIR,
                                experiment_group=group_dir_rel,
                                modality=MODALITIES[mod_num],
                                gan_suffix=gan_suffix,
                                n_epochs=N_EPOCHS,
                                **pre_params["normalization_parameters"],
                            )
                        )
                # see if all paths exist, this is important, if two processes
                #  are preparing the data
                if not np.all([(experiment_dir / p).exists() for p in model_paths]):
                    continue
                pre_params["normalization_parameters"]["model_paths"] = tuple(model_paths)
                # overlap has already been cut
                CUT_TO_OVERLAP = False
                preprocess_base_dir = experiment_dir
            else:
                PASS_MODALITY = False
                CUT_TO_OVERLAP = True
                DATASET_TO_PROCESS = hyp_dataset
                preprocess_base_dir = data_dir
                preprocessing_name = norm_type.name

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

            if hyp["architecture"] is ResNet:
                tasks = ("classification", "regression")
                fold_dir = group_dir_rel / "folds_class_reg"
            else:
                tasks = ("segmentation",)
                fold_dir = group_dir_rel / "folds"

            # make sure that there are samples for all labels
            for tsk in ["classification", "regression"]:
                if tsk not in tasks:
                    continue
                tsk_df = pd.DataFrame([exp_dataset[t][tsk] for t in train_list])
                n_none = tsk_df.isna().mean()
                for to_delete in n_none[n_none > 0.9].index:
                    print(
                        f"Only missing values found in {to_delete}."
                        + " It will not be used for training."
                    )
                    for val in exp_dataset.values():
                        if to_delete in val[tsk]:
                            del val[tsk][to_delete]

            exp = Experiment(
                hyper_parameters=hyp,
                name=experiment_name,
                output_path_rel=group_dir_rel / experiment_name,
                data_set=exp_dataset,
                crossvalidation_set=train_list,
                external_test_set=test_list,
                tasks=tasks,
                folds=K_FOLD,
                seed=42,
                num_channels=N_CHANNELS,
                folds_dir_rel=fold_dir,
                tensorboard_images=True,
                priority=priority(hyp, location),
            )
            experiments.append(exp)
            assert preprocessing_name in experiment_name
            additional_info.append({"normalization": preprocessing_name})

            # bring experiments in a custom order
            experiments_sorted = sorted(experiments, key=lambda x: x.priority, reverse=True)
            # export all hyperparameters
            export_experiments_run_files(
                exp_group_base_dir, experiments, additional_info=additional_info
            )

            print(f"Exported {location} - {experiment_name}")
