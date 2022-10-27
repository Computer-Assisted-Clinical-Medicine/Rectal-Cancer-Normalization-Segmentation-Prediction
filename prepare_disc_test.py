"""
Prepare the training and run it on the cluster or a local machine (automatically detected)
"""
import logging
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# logger has to be set before tensorflow is imported
tf_logger = logging.getLogger("tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

from gan_normalization import train_gan_normalization
from SegClassRegBasis.utils import get_gpu

# set tf thread mode
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

### normalization method ###
preprocessing_parameters: List[Tuple[str, dict]] = [
    (
        "",
        {
            "depth": 3,
            "filter_base": 16,
            "min_max": False,
            "smoothing_sigma": 1,
            "latent_weight": 1,
            "image_weight": 1,
            "skip_edges": True,
            "latent": False,
            "train_on_gen": False,
            "n_epochs": 50,
            "batch_size": 64,
        },
    ),
    (
        "_f16",
        {
            "depth": 3,
            "filter_base": 16,
            "min_max": False,
            "smoothing_sigma": 1,
            "latent_weight": 1,
            "image_weight": 1,
            "disc_filter_base": 16,
            "skip_edges": True,
            "latent": False,
            "train_on_gen": False,
            "n_epochs": 50,
            "batch_size": 64,
        },
    ),
    (
        "_f8",
        {
            "depth": 3,
            "filter_base": 16,
            "min_max": False,
            "smoothing_sigma": 1,
            "latent_weight": 1,
            "image_weight": 1,
            "disc_filter_base": 16,
            "skip_edges": True,
            "latent": False,
            "train_on_gen": False,
            "n_epochs": 50,
            "batch_size": 64,
        },
    ),
    (
        "_BetterConv",
        {
            "depth": 3,
            "filter_base": 64,
            "min_max": False,
            "smoothing_sigma": 0.5,
            "latent_weight": 1,
            "image_weight": 1,
            "image_gen_weight": 0.5,
            "skip_edges": True,
            "latent": False,
            "train_on_gen": False,
            "disc_type": "BetterConv",
            "n_epochs": 50,
            "batch_size": 32,
        },
    ),
    (
        "_BetterConv",
        {
            "depth": 3,
            "filter_base": 64,
            "min_max": False,
            "smoothing_sigma": 0.5,
            "latent_weight": 1,
            "image_weight": 1,
            "image_gen_weight": 0.5,
            "skip_edges": True,
            "latent": False,
            "train_on_gen": False,
            "disc_type": "BetterConv",
            "n_epochs": 50,
            "batch_size": 32,
            "disc_start_lr": 0.05,
            "disc_end_lr": 0.001,
        },
    ),
    (
        "_BetterConv_lr_em2",
        {
            "depth": 3,
            "filter_base": 64,
            "min_max": False,
            "smoothing_sigma": 0.5,
            "latent_weight": 1,
            "image_weight": 1,
            "image_gen_weight": 0.5,
            "skip_edges": True,
            "latent": False,
            "train_on_gen": False,
            "disc_type": "BetterConv",
            "n_epochs": 50,
            "batch_size": 32,
            "disc_start_lr": 1e-2,
            "disc_end_lr": 1e-2,
        },
    ),
    (
        "_BetterConv_lr_em3",
        {
            "depth": 3,
            "filter_base": 64,
            "min_max": False,
            "smoothing_sigma": 0.5,
            "latent_weight": 1,
            "image_weight": 1,
            "image_gen_weight": 0.5,
            "skip_edges": True,
            "latent": False,
            "train_on_gen": False,
            "disc_type": "BetterConv",
            "n_epochs": 50,
            "batch_size": 32,
            "disc_start_lr": 1e-3,
            "disc_end_lr": 1e-3,
        },
    ),
    (
        "_BetterConv_lr_em4",
        {
            "depth": 3,
            "filter_base": 64,
            "min_max": False,
            "smoothing_sigma": 0.5,
            "latent_weight": 1,
            "image_weight": 1,
            "image_gen_weight": 0.5,
            "skip_edges": True,
            "latent": False,
            "train_on_gen": False,
            "disc_type": "BetterConv",
            "n_epochs": 50,
            "batch_size": 32,
            "disc_start_lr": 1e-4,
            "disc_end_lr": 1e-4,
        },
    ),
    (
        "_BetterConv_lr_em5",
        {
            "depth": 3,
            "filter_base": 64,
            "min_max": False,
            "smoothing_sigma": 0.5,
            "latent_weight": 1,
            "image_weight": 1,
            "image_gen_weight": 0.5,
            "skip_edges": True,
            "latent": False,
            "train_on_gen": False,
            "disc_type": "BetterConv",
            "n_epochs": 50,
            "batch_size": 32,
            "disc_start_lr": 1e-5,
            "disc_end_lr": 1e-5,
        },
    ),
    (
        "_BetterConv_lr_em6",
        {
            "depth": 3,
            "filter_base": 64,
            "min_max": False,
            "smoothing_sigma": 0.5,
            "latent_weight": 1,
            "image_weight": 1,
            "image_gen_weight": 0.5,
            "skip_edges": True,
            "latent": False,
            "train_on_gen": False,
            "disc_type": "BetterConv",
            "n_epochs": 50,
            "batch_size": 32,
            "disc_start_lr": 1e-6,
            "disc_end_lr": 1e-6,
        },
    ),
]

if __name__ == "__main__":

    data_dir = Path(os.environ["data_dir"])
    experiment_dir = Path(os.environ["experiment_dir"])
    GROUP_BASE_NAME = "Discriminator_Test"
    exp_group_base_dir = experiment_dir / GROUP_BASE_NAME
    if not exp_group_base_dir.exists():
        exp_group_base_dir.mkdir(parents=True)

    gpu = tf.device(get_gpu(memory_limit=2000))

    # load data
    timepoints = pd.read_csv(data_dir / "timepoints.csv", sep=";", index_col=0)

    N_CHANNELS = 3
    MODALITIES = ("T2 axial", "Diff axial b800", "ADC axial original")

    # set config
    PREPROCESSED_DIR = Path(GROUP_BASE_NAME)

    # only use before therapy images that are segmented
    timepoints_train = timepoints.query(
        "treatment_status=='before therapy' & segmented & 'Frankfurt' in location"
    ).index
    timepoints_train_norm = timepoints.query("'Frankfurt' in location").index

    group_dir_rel = Path(GROUP_BASE_NAME)

    for mod_num in range(N_CHANNELS):
        for gan_suffix, norm_params in preprocessing_parameters:
            with gpu:
                train_gan_normalization(
                    timepoints_train=timepoints_train_norm,
                    mod_num=mod_num,
                    preprocessed_dir=PREPROCESSED_DIR,
                    experiment_group=group_dir_rel,
                    modality=MODALITIES[mod_num],
                    gan_suffix=gan_suffix,
                    identity=True,
                    **norm_params,
                )
