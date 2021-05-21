'''
Run the training and evaluation of the models.
'''
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

#logger has to be set before tensorflow is imported
tf_logger = logging.getLogger('tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# pylint: disable=wrong-import-position

from experiment import Experiment, export_batch_file
from SegmentationNetworkBasis.architecture import UNet
from SegmentationNetworkBasis.segbasisloader import NORMALIZING
from utils import compare_hyperparameters, plot_hparam_comparison, generate_folder_name

# set tf thread mode
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

DEBUG = False
if DEBUG:
    # run everything eagerly
    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(True)
    # do numeric checks (for NaNs)
    tf.debugging.enable_check_numerics(
        stack_height_limit=30, path_length_limit=50
    )


if __name__ == '__main__':

    #configure loggers
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    tf_logger.setLevel(logging.DEBUG)
    #there is too much output otherwise
    for h in tf_logger.handlers:
        tf_logger.removeHandler(h)


    data_dir = Path(os.environ['data_dir'])
    experiment_dir = Path(os.environ['experiment_dir'])
    if not experiment_dir.exists():
        experiment_dir.mkdir()

    #configure logging to only log errors
    #create file handler
    fh = logging.FileHandler(experiment_dir/'log_errors.txt')
    fh.setLevel(logging.ERROR)
    # create formatter
    formatter = logging.Formatter(
        '%(levelname)s: %(name)s - %(funcName)s (l.%(lineno)d): %(message)s'
    )
    # add formatter to fh
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    #print errors (also for tensorflow)
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.ERROR)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    tf_logger.addHandler(ch)

    # load training files
    train_list = np.loadtxt(data_dir / 'train_IDs.csv', dtype='str')
    train_list = np.array([str(t) for t in train_list])

    # load test files
    test_list = np.loadtxt(data_dir / 'test_IDs.csv', dtype='str')
    test_list = np.array([str(t) for t in test_list])

    K_FOLD = 5

    # get number of channels from database file
    data_dict_file = data_dir / 'dataset.json'
    if not data_dict_file.exists():
        raise FileNotFoundError(f'Dataset dict file {data_dict_file} not found.')
    with open(data_dict_file) as f:
        data_dict = json.load(f)
    n_channels = len(data_dict['modality'])

    #define the parameters that are constant
    F_BASE = 8
    init_parameters = {
        "regularize": [True, 'L2', 1e-5],
        "drop_out": [True, 0.01],
        "activation": "elu",
        "do_bias": True,
        "cross_hair": False,
        "clipping_value" : 50,
        "res_connect" : True,
        'n_filters' : [F_BASE*8, F_BASE*16, F_BASE*32, F_BASE*64, F_BASE*128]
    }

    train_parameters = {
        "l_r" : 0.001,
        "optimizer" : "Adam",
        "epochs" : 100,
        # scheduling parameters
        "early_stopping" : True,
        "patience_es" : 15,
        "reduce_lr_on_plateau" : True,
        "patience_lr_plat" : 5,
        "factor_lr_plat" : 0.5,
        # sampling parameters
        "percent_of_object_samples" : 0.33,
        "samples_per_volume" : 80,
        "background_label_percentage" : 0.15,
        # Augmentation parameters
        "add_noise" : False,
        "max_rotation" : 0,
        "min_resolution_augment" : 1,
        "max_resolution_augment" : 1
    }

    data_loader_parameters = {
        "do_resampling" : True
    }

    constant_parameters = {
        "init_parameters" : init_parameters,
        "train_parameters" : train_parameters,
        "data_loader_parameters" : data_loader_parameters,
        "loss" : 'DICE',
        'architecture' : UNet
    }

    # normalization method
    normalization_methods = [
        NORMALIZING.HM_QUANTILE, # NORMALIZING.HM_QUANT_MEAN,
        # NORMALIZING.HISTOGRAM_MATCHING, NORMALIZING.Z_SCORE,
        NORMALIZING.QUANTILE, NORMALIZING.MEAN_STD
    ]
    # do batch norm
    batch_norm = [False]
    # dimensions
    dimensions = [2, 3]

    #generate tensorflow command
    tensorboard_command = f'tensorboard --logdir="{experiment_dir.resolve()}"'
    print(f'To see the progress in tensorboard, run:\n{tensorboard_command}')

    # set config
    PREPROCESSED_DIR = 'data_preprocessed'

    #set up all experiments
    experiments = []
    for d in dimensions:
        for n in normalization_methods:
            for b in batch_norm:
                hyper_parameters : Dict[str, Any] = {
                    **constant_parameters,
                    'dimensions' : d
                }
                hyper_parameters['init_parameters']['do_batch_normalization'] = b
                # bias should be the opposite of batch norm
                hyper_parameters['init_parameters']['do_bias'] = not b
                hyper_parameters['data_loader_parameters']['normalizing_method'] = n         

                # use less filters for 3D on local computer
                if not 'CLUSTER' in os.environ:
                    if d == 3:
                        F_BASE = 4
                        n_filters = [F_BASE*8, F_BASE*16, F_BASE*32, F_BASE*64, F_BASE*128]
                        hyper_parameters['init_parameters']['n_filters'] = n_filters
                    else:
                        F_BASE = 8
                        n_filters = [F_BASE*8, F_BASE*16, F_BASE*32, F_BASE*64, F_BASE*128]
                        hyper_parameters['init_parameters']['n_filters'] = n_filters

                #define experiment
                experiment_name = generate_folder_name(hyper_parameters)  # pylint: disable=invalid-name

                experiment = Experiment(
                    hyper_parameters=hyper_parameters,
                    name=experiment_name,
                    output_path_rel=experiment_name,
                    data_set=train_list,
                    external_test_set=test_list,
                    folds=K_FOLD,
                    seed=42,
                    num_channels=n_channels,
                    folds_dir_rel='folds',
                    preprocessed_dir_rel=PREPROCESSED_DIR,
                    tensorboard_images=True
                )
                experiments.append(experiment)

    # export all hyperparameters
    compare_hyperparameters(experiments, experiment_dir)

    # if on cluster, export slurm files
    if 'CLUSTER' in os.environ:
        slurm_files = []
        working_dir = Path('').resolve()
        if not working_dir.exists():
            working_dir.mkdir()
        for e in experiments:
            slurm_files.append(e.export_slurm_file(working_dir))

        export_batch_file(
            filename=experiment_dir / 'start_all_jobs.sh',
            commands=[f'sbatch {f}' for f in slurm_files]
        )

        # and create some needed directories (without their log dirs, jobs don't start)
        plot_dir_slurm = working_dir / 'plots' / 'slurm'
        if not plot_dir_slurm.exists():
            plot_dir_slurm.mkdir(parents=True)
        combined_dir_slurm = working_dir / 'combined_models' / 'slurm'
        if not combined_dir_slurm.exists():
            combined_dir_slurm.mkdir(parents=True)
        sys.exit()

    # if not on cluster, perform the experiments
    for e in experiments:
        # run all folds
        for fold_num in range(K_FOLD):
            #add more detailed logger for each network, when problems arise, use debug
            fold_dir = e.output_path / e.fold_dir_names[fold_num]
            if not fold_dir.exists():
                fold_dir.mkdir(parents=True)

            #create file handlers
            fh_info = logging.FileHandler(fold_dir/'log_info.txt')
            fh_info.setLevel(logging.INFO)
            fh_info.setFormatter(formatter)
            #add to loggers
            logger.addHandler(fh_info)

            #create file handlers
            fh_debug = logging.FileHandler(fold_dir/'log_debug.txt')
            fh_debug.setLevel(logging.DEBUG)
            fh_debug.setFormatter(formatter)
            #add to loggers
            logger.addHandler(fh_debug)

            try:
                e.run_fold(fold_num)
            except Exception as exc: # pylint: disable=broad-except
                logging.exception(str(exc))
                print(exc)
                print('Training failed')
                # remove tensorboard log dir if training failed (to not clutter tensorboard)
                tb_log_dir = fold_dir / 'logs'
                if tb_log_dir.exists():
                    shutil.rmtree(tb_log_dir)

            #remove logger
            logger.removeHandler(fh_info)
            logger.removeHandler(fh_debug)

        # evaluate all experiments
        e.evaluate()
        # also evaluate on the external testset
        e.evaluate_external_testset()
        # do intermediate plots (at least try to)
        try:
            plot_hparam_comparison(experiment_dir)
            plot_hparam_comparison(experiment_dir, external=True)
            plot_hparam_comparison(experiment_dir, postprocessed=True)
            plot_hparam_comparison(experiment_dir, external=True, postprocessed=True)
        except Exception as exc: # pylint: disable=broad-except
            # log the error
            logging.exception(str(exc))
            print(f'Failed to to intermediate plots because of {e}.')
