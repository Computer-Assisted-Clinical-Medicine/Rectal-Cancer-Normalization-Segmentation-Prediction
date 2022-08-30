"""
Test an architecture, it is only checked for errors and dimensions. The
functionality is not evaluated.
"""
import tempfile
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf

import seg_data_loader
from SegClassRegBasis import architecture
from SegClassRegBasis import config as cfg
from SegClassRegBasis.test_seg_data_loader import (
    get_loader,
    load_dataset,
    set_parameters_according_to_dimension,
    set_seeds,
)

if __name__ == "__main__":
    test_dir = Path("test_data")

    DEBUG = False  # some behavior might change
    DIMENSION = 2
    N_EPOCHS = 100
    NETWORK = architecture.DeepLabv3plus
    PREPROCESSED_DIR = test_dir / f"{cfg.num_channels}_channels" / "data_preprocessed"
    NUM_CHANNELS = 3

    cfg.num_channels = NUM_CHANNELS

    if DEBUG:
        tf.debugging.enable_check_numerics(stack_height_limit=60, path_length_limit=100)

    F_BASE = 8
    # define the parameters that are constant
    train_parameters = {
        "l_r": 0.001,
        "optimizer": "Adam",
        "epochs": N_EPOCHS,
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

    constant_parameters = {
        "train_parameters": train_parameters,
        "loss": "DICE",
    }
    # define constant parameters
    hyper_parameters: Dict[str, Any] = {
        **constant_parameters,
    }

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

    hyper_parameters["network_parameters"] = network_parameters_DenseTiramisu
    hyper_parameters["dimensions"] = DIMENSION

    hyper_parameters["train_parameters"]["percent_of_object_samples"] = 0.4

    set_parameters_according_to_dimension(
        DIMENSION, NUM_CHANNELS, PREPROCESSED_DIR, NETWORK.get_name()
    )

    set_seeds()

    # generate loader
    data_loader = get_loader("train", seg_data_loader)

    file_list, _, file_dict = load_dataset(test_dir)

    cfg.num_files = len(file_list) - cfg.number_of_vald

    train_dataset = data_loader(
        file_list[: -cfg.number_of_vald],
        batch_size=cfg.batch_size_train,
        n_epochs=N_EPOCHS,
        read_threads=cfg.train_reader_instances,
        file_dict=file_dict,
    )
    valid_dataset = data_loader(
        file_list[-cfg.number_of_vald :],
        batch_size=cfg.batch_size_train,
        n_epochs=N_EPOCHS,
        read_threads=cfg.train_reader_instances,
        file_dict=file_dict,
    )
    visualization_dataset = data_loader(
        file_list[:1],
        batch_size=cfg.batch_size_train,
        n_epochs=N_EPOCHS,
        read_threads=cfg.train_reader_instances,
    )

    # do training
    with tempfile.TemporaryDirectory() as tempdir, tf.device("/device:GPU:0"):

        fold_dir = Path(tempdir) / "fold-0"
        if not (fold_dir).exists():
            fold_dir.mkdir()

        net = NETWORK(
            hyper_parameters["loss"],
            debug=DEBUG,
            # add initialization parameters
            **hyper_parameters["network_parameters"],
        )

        net.model.summary(line_length=150)

        # generate tensorflow command
        tensorboard_command = f'tensorboard --logdir="{tempdir}"'
        print(f"To see the progress in tensorboard, run:\n{tensorboard_command}")

        net.train(
            logs_path=tempdir,
            folder_name=fold_dir.name,
            training_dataset=train_dataset,
            validation_dataset=valid_dataset,
            visualization_dataset=visualization_dataset,
            write_graph=True,
            debug=DEBUG,
            # add training parameters
            **(hyper_parameters["train_parameters"]),
        )
