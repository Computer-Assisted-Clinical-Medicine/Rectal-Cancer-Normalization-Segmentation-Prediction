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

from experiment import Experiment, export_batch_file
from SegmentationNetworkBasis.architecture import UNet
from SegmentationNetworkBasis.segbasisloader import NORMALIZING


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


def plot_hparam_comparison(experiment_dir, metrics = ['Dice']):
    hparam_file = experiment_dir / 'hyperparameters.csv'
    hparam_changed_file = experiment_dir / 'hyperparameters_changed.csv'

    hparams = pd.read_csv(hparam_file)
    hparams_changed = pd.read_csv(hparam_changed_file)
    changed_params = hparams_changed.columns[1:]
    # collect all results
    results_means = []
    results_stds = []
    for results_file in hparams['result_file']:
        if Path(results_file).exists():
            results = pd.read_csv(results_file)
            # save results
            results_means.append(results[metrics].mean())
            results_stds.append(results[metrics].std())
        else:
            print(f'Could not find the evaluation file {results_file}.')
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
            for group, data in hparams_changed.groupby(unused_columns):
                # plot them with the same line
                # get the data
                m_data = results_means.loc[data.index,m]
                # only plot if not nan
                if not m_data.isna().all():
                    ax.plot(data.loc[m_data.notna(), c], m_data[m_data.notna()], marker='x', label=str(group))
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
    plt.savefig(experiment_dir / 'hyperparameter_comparison.pdf')
    plt.close()


def compare_hyperparameters(experiments, experiment_dir):
    # export the hyperparameters
    hyperparameter_file = experiment_dir / 'hyperparameters.csv'
    hyperparameter_changed_file = experiment_dir / 'hyperparameters_changed.csv'
    # collect all results
    hparams = []
    for e in experiments:
        results_file = e.output_path / 'evaluation-all-files.csv'
        # and parameters
        hparams.append({
            **e.hyper_parameters['init_parameters'],
            **e.hyper_parameters['train_parameters'],
            **e.hyper_parameters['data_loader_parameters'],
            'loss' : e.hyper_parameters['loss'],
            'architecture' : e.hyper_parameters['architecture'].__name__,
            'dimensions' : e.hyper_parameters['dimensions'],
            'result_file' : results_file
        })

    # convert to dataframes
    hparams = pd.DataFrame(hparams)
    # find changed parameters
    changed_params = []
    # drop the results file when analyzing the changed hyperparameters
    for c in hparams.drop(columns='result_file'):
        if hparams[c].astype(str).unique().size > 1:
            changed_params.append(c)
    hparams_changed = hparams[changed_params].copy()
    # if n_filters, use the first
    if 'n_filters' in hparams_changed:
        hparams_changed.loc[:,'n_filters'] = hparams_changed['n_filters'].apply(lambda x: x[0])
    if 'normalizing_method' in hparams_changed:
        hparams_changed.loc[:,'normalizing_method'] = hparams_changed['normalizing_method'].apply(lambda x: x.name)

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

    train_list = np.loadtxt(data_dir / 'train_IDs.csv', dtype='str')

    data_list = np.array([str(data_dir / t) for t in train_list])

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
        "normalizing_method" : NORMALIZING.MEAN_STD,
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
    normalization_methods = [NORMALIZING.QUANTILE, NORMALIZING.MEAN_STD]
    # do batch norm
    batch_norm = [True, False]
    # dimensions
    dimensions = [2, 3]

    #generate tensorflow command
    tensorboard_command = f'tensorboard --logdir="{experiment_dir.absolute()}"'
    print(f'To see the progress in tensorboard, run:\n{tensorboard_command}')

    # set config
    preprocessed_dir = experiment_dir / 'data_preprocessed'
    if not preprocessed_dir.exists():
        preprocessed_dir.mkdir()

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
                hyper_parameters['data_loader_parameters']['normalizing_method'] = n

                #define experiment
                experiment_name = generate_folder_name(hyper_parameters)
                
                current_experiment_path = Path(experiment_dir, experiment_name)
                if not current_experiment_path.exists():
                    current_experiment_path.mkdir()

                experiment = Experiment(
                    hyper_parameters=hyper_parameters,
                    name=experiment_name,
                    output_path=current_experiment_path,
                    data_set=data_list,
                    folds=k_fold,
                    num_channels=n_channels,
                    folds_dir=experiment_dir / 'folds',
                    preprocessed_dir=preprocessed_dir,
                    tensorboard_images=True
                )
                experiments.append(experiment)

    # export all hyperparameters
    compare_hyperparameters(experiments, experiment_dir)

    # if on cluster, export slurm files
    if 'CLUSTER' in os.environ:
        slurm_files = []
        working_dir = Path('').absolute()
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
        # do intermediate plots
        plot_hparam_comparison(experiment_dir)