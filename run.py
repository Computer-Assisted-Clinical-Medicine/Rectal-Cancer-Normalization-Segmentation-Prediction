import json
import logging
import os
import shutil
import sys
from pathlib import Path

import matplotlib

# if on cluster, use other backend
if 'CLUSTER' in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#logger has to be set before tensorflow is imported
tf_logger = logging.getLogger('tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# set tf thread mode
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

from experiment import Experiment, export_batch_file
from SegmentationNetworkBasis.architecture import UNet
from SegmentationNetworkBasis.segbasisloader import NORMALIZING

debug = False
if debug:
    # run everything eagerly
    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(True)
    # do numeric checks (for NaNs)
    tf.debugging.enable_check_numerics(
        stack_height_limit=30, path_length_limit=50
    )


def generate_folder_name(hyper_parameters):
    epochs = hyper_parameters['train_parameters']['epochs']

    params = [
        hyper_parameters['architecture'].get_name() + str(hyper_parameters['dimensions']) + 'D',
        hyper_parameters['loss']
    ]

    # residual connections if it is an attribute
    if 'res_connect' in hyper_parameters['init_parameters']:
        if hyper_parameters['init_parameters']['res_connect']:
            params.append('Res')
        else:
            params.append('nRes')

    # filter multiplier
    params.append('f_'+str(hyper_parameters['init_parameters']['n_filters'][0]//8))

    # batch norm
    if hyper_parameters['init_parameters']['do_batch_normalization']:
        params.append('BN')
    else:
        params.append('nBN')

    # dropout
    if hyper_parameters['init_parameters']['drop_out'][0]:
        params.append('DO')
    else:
        params.append('nDO')

    # normalization
    params.append(str(hyper_parameters['data_loader_parameters']['normalizing_method'].name))

    # add epoch number
    params.append(str(epochs))

    folder_name = "-".join(params)

    return folder_name


def plot_hparam_comparison(experiment_dir, metrics = ['Dice'], external=False, postprocessed=False):
    hparam_file = experiment_dir / 'hyperparameters.csv'
    hparam_changed_file = experiment_dir / 'hyperparameters_changed.csv'

    if external:
        file_field = 'results_file_external_testset'
        result_name = 'hyperparameter_comparison_external_testset'
    else:
        file_field = 'results_file'
        result_name = 'hyperparameter_comparison'

    if postprocessed:
        file_field += '_postprocessed'
        result_name += '_postprocessed'

    # add pdf
    result_name += '.pdf'

    hparams = pd.read_csv(hparam_file)
    hparams_changed = pd.read_csv(hparam_changed_file)
    changed_params = hparams_changed.columns[1:]
    # collect all results
    results_means = []
    results_stds = []
    for results_file in hparams[file_field]:
        if Path(results_file).exists():
            results = pd.read_csv(results_file)
            # save results
            results_means.append(results[metrics].mean())
            results_stds.append(results[metrics].std())
        else:
            print(f'Could not find the evaluation file {results_file}'
                +' (probably not finished with training yet).')
            results_means.append(pd.Series({m : pd.NA for m in metrics}))
            results_stds.append(pd.Series({m : pd.NA for m in metrics}))

    # convert to dataframes
    results_means = pd.DataFrame(results_means)
    results_stds = pd.DataFrame(results_stds)

    # plot all metrics with all parameters
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(changed_params),
        sharey=True,
        figsize=(4*len(changed_params),6*len(metrics))
    )
    # fix the dimensions
    axes = np.array(axes).reshape((len(metrics), len(changed_params)))

    for m, ax_row in zip(metrics, axes):
        for c, ax in zip(changed_params, ax_row):
            # group by the other values
            unused_columns = [cn for cn in changed_params if c != cn]
            # if there are no unused columns, use the changed one
            if len(unused_columns) == 0:
                unused_columns = list(changed_params)
            for group, data in hparams_changed.groupby(unused_columns):
                # plot them with the same line
                # get the data
                m_data = results_means.loc[data.index,m]
                # sort by values
                m_data.sort_values()
                # only plot if not nan
                if not m_data.isna().all():
                    ax.plot(
                        data.loc[m_data.notna(), c], m_data[m_data.notna()],
                        marker='x',
                        label=str(group)
                    )
            # if the label is text, turn it
            labels = hparams_changed[c]
            if not pd.api.types.is_numeric_dtype(labels):
                ax.set_xticks(list(labels))
                ax.set_xticklabels(list(labels), rotation=45, ha='right')
            # ylabel if it is the first image
            if c == changed_params[0]:
                ax.set_ylabel(m)
            # xlabel if it is the last row
            if m == metrics[-1]:
                ax.set_xlabel(c)
            # if the class is bool, replace the labels with the boolean values
            if type(hparams_changed.iloc[0][c]) == np.bool_:
                ax.set_xticks([0,1])
                ax.set_xticklabels(['false', 'true'])

            # set the legend with title
            ax.legend(title = str(tuple(str(c)[:5] for c in unused_columns)))

    fig.suptitle('Hypereparameter Comparison')
    plt.tight_layout()
    plt.savefig(experiment_dir / result_name)
    plt.close()


def compare_hyperparameters(experiments, experiment_dir):
    # export the hyperparameters
    hyperparameter_file = experiment_dir / 'hyperparameters.csv'
    hyperparameter_changed_file = experiment_dir / 'hyperparameters_changed.csv'
    # collect all results
    hparams = []
    for e in experiments:
        res_name = 'evaluation-all-files.csv'
        results_file = e.output_path / 'results_test_final' / res_name
        results_file_postprocessed = e.output_path / 'results_test_final-postprocessed' / res_name
        results_file_external = e.output_path / 'results_external_testset_final' / res_name
        results_file_external_testset_postprocessed = e.output_path / 'results_external_testset_final-postprocessed' / res_name
        # and parameters
        hparams.append({
            **e.hyper_parameters['init_parameters'],
            **e.hyper_parameters['train_parameters'],
            **e.hyper_parameters['data_loader_parameters'],
            'loss' : e.hyper_parameters['loss'],
            'architecture' : e.hyper_parameters['architecture'].__name__,
            'dimensions' : e.hyper_parameters['dimensions'],
            'path' : e.output_path,
            'results_file' : results_file,
            'results_file_postprocessed' : results_file_postprocessed,
            'results_file_external_testset' : results_file_external,
            'results_file_external_testset_postprocessed' : results_file_external_testset_postprocessed
        })

    # convert to dataframes
    hparams = pd.DataFrame(hparams)
    # find changed parameters
    changed_params = []
    # drop the results file when analyzing the changed hyperparameters
    for c in hparams.drop(columns='results_file'):
        if hparams[c].astype(str).unique().size > 1:
            changed_params.append(c)
    hparams_changed = hparams[changed_params].copy()
    # if n_filters, use the first
    if 'n_filters' in hparams_changed:
        hparams_changed.loc[:,'n_filters'] = hparams_changed['n_filters'].apply(lambda x: x[0])
    if 'normalizing_method' in hparams_changed:
        hparams_changed.loc[:,'normalizing_method'] = hparams_changed['normalizing_method'].apply(lambda x: x.name)
    # ignore do_bias (it is set the opposite to batch_norm)
    if 'do_bias' in hparams_changed and 'do_batch_normalization' in hparams_changed:
        hparams_changed.drop(columns='do_bias', inplace=True)
    # drop column specifying the files
    if 'path' in hparams_changed:
        hparams_changed.drop(columns='path', inplace=True)
    # drop column specifying the files
    if 'results_file_postprocessed' in hparams_changed:
        hparams_changed.drop(columns='results_file_postprocessed', inplace=True)
    # drop column specifying the files
    if 'results_file_external_testset' in hparams_changed:
        hparams_changed.drop(columns='results_file_external_testset', inplace=True)
    # drop column specifying the files
    if 'results_file_external_testset_postprocessed' in hparams_changed:
        hparams_changed.drop(columns='results_file_external_testset_postprocessed', inplace=True)

    hparams.to_csv(hyperparameter_file)
    hparams_changed.to_csv(hyperparameter_changed_file)


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
    formatter = logging.Formatter('%(levelname)s: %(name)s - %(funcName)s (l.%(lineno)d): %(message)s')
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

    k_fold = 5

    # get number of channels from database file
    data_dict_file = data_dir / 'dataset.json'
    if not data_dict_file.exists():
        raise FileNotFoundError(f'Dataset dict file {data_dict_file} not found.')
    with open(data_dict_file) as f:
        data_dict = json.load(f)
    n_channels = len(data_dict['modality'])

    #define the parameters that are constant
    f = 8
    init_parameters = {
        "regularize": [True, 'L2', 0.0000001],
        "drop_out": [True, 0.01],
        "activation": "elu",
        "do_bias": True,
        "cross_hair": False,
        "clipping_value" : 50,
        "res_connect" : True,
        'n_filters' : [f*8, f*16, f*32, f*64, f*128]
    }

    train_parameters = {
        "l_r" : 0.001,
        "optimizer" : "Adam",
        "epochs" : 100,
        "early_stopping" : True,
        "patience_es" : 15,
        "reduce_lr_on_plateau" : True,
        "patience_lr_plat" : 5,
        "factor_lr_plat" : 0.5
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
        NORMALIZING.HM_QUANTILE, NORMALIZING.HM_QUANT_MEAN,
        NORMALIZING.HISTOGRAM_MATCHING, NORMALIZING.Z_SCORE,
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
    preprocessed_dir = 'data_preprocessed'

    #set up all experiments
    experiments = []
    for n in normalization_methods:
        for b in batch_norm:
            for d in dimensions:
                hyper_parameters = {
                    **constant_parameters,
                    'dimensions' : d
                }
                hyper_parameters['init_parameters']['do_batch_normalization'] = b
                hyper_parameters['init_parameters']['do_bias'] = not b # bias should be the opposite of batch norm
                hyper_parameters['data_loader_parameters']['normalizing_method'] = n

                #define experiment
                experiment_name = generate_folder_name(hyper_parameters)
                
                experiment = Experiment(
                    hyper_parameters=hyper_parameters,
                    name=experiment_name,
                    output_path_rel=experiment_name,
                    data_set=train_list,
                    external_test_set=test_list,
                    folds=k_fold,
                    num_channels=n_channels,
                    folds_dir_rel='folds',
                    preprocessed_dir_rel=preprocessed_dir,
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
        sys.exit()

    # if not on cluster, perform the experiments
    for e in experiments:
        # run all folds
        for f in range(k_fold):
            #add more detailed logger for each network, when problems arise, use debug
            fold_dir = e.output_path / e.fold_dir_names[f]
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
                e.run_fold(f)
            except Exception as e:
                print(e)
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
        except Exception as e:
            print(f'Failed to to intermediate plots because of {e}.')