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
init_parameters = {
    "regularize": [True, 'L2', 0.0000001],
    "drop_out": [True, 0.01],
    "activation": "elu",
    "do_batch_normalization": False,
    "do_bias": True,
    "cross_hair": False,
    "clipping_value" : 50,
}

train_parameters = {
    "l_r": 0.001,
    "optimizer": "Adam",
    "epochs" : 100,
    "early_stopping" : True,
    "patience_es" : 15,
    "reduce_lr_on_plateau" : True,
    "patience_lr_plat" : 5,
    "factor_lr_plat" : 0.5
}

constant_parameters = {
    "init_parameters": init_parameters,
    "train_parameters": train_parameters,
    "loss" : 'DICE',
    'architecture' : UNet
}

#define the parameters that are being tuned
dimensions = [2, 3]
# residual connections
res = [True, False]
# multiply the number of filters by a fixed factor
filter_multiplier = [1, 2, 4, 8]

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

    # add epoch number
    params.append(str(epochs))

    folder_name = "-".join(params)

    return folder_name

#generate tensorflow command
tensorboard_command = f'tensorboard --logdir="{experiment_dir.absolute()}"'
print(f'To see the progress in tensorboard, run:\n{tensorboard_command}')

# set config
preprocessed_dir = Path(os.environ['experiment_dir']) / 'data_preprocessed'
if not preprocessed_dir.exists():
    preprocessed_dir.mkdir()

#set up all experiments
experiments = []
for d in dimensions:
    for f in filter_multiplier:
        for r in res:
            hyper_parameters = {
                **constant_parameters,
                'dimensions' : d,
            }
            hyper_parameters['init_parameters']['res_connect'] = r
            hyper_parameters['init_parameters']['n_filters'] = [f*8, f*16, f*32, f*64, f*128]

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

# if on cluster, export slurm files
if 'CLUSTER' in os.environ:
    slurm_files = []
    working_dir = Path('').absolute()
    if not working_dir.exists():
        working_dir.mkdir()
    for e in experiments:
        slurm_files.append(e.export_slurm_file(working_dir))

    export_batch_file(
        filename=Path(os.environ['experiment_dir']) / 'start_all_jobs.sh',
        commands=[f'sbatch {f}' for f in slurm_files]
    )
    sys.exit()


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

# collect all results
metrics = ['Dice']
results_means = []
results_stds = []
hparams = []
for e in experiments:
    results_file = e.output_path / 'evaluation-all-files.csv'
    if results_file.exists():
        results = pd.read_csv(results_file)
        # save results
        results_means.append(results[metrics].mean())
        results_stds.append(results[metrics].std())
        # and parameters
        hparams.append({
            **e.hyper_parameters['init_parameters'],
            **e.hyper_parameters['train_parameters'],
            'loss' : e.hyper_parameters['loss'],
            'architecture' : e.hyper_parameters['architecture'],
            'dimensions' : e.hyper_parameters['dimensions']
        })
    else:
        print(f'Could not find the evaluation file {results_file}.')

# convert to dataframes
hparams = pd.DataFrame(hparams)
results_means = pd.DataFrame(results_means)
results_stds = pd.DataFrame(results_stds)
# find changed parameters
changed_params = []
for c in hparams:
    if hparams[c].astype(str).unique().size > 1:
        changed_params.append(c)
hparams_changed = hparams[changed_params]
# if n_filters, use the first
if 'n_filters' in hparams_changed:
    hparams_changed.loc[hparams_changed.index,'n_filters'] = hparams_changed['n_filters'].apply(lambda x: x[0])

# plot all metrics with all parameters
fig, axes = plt.subplots(nrows=len(metrics), ncols=len(changed_params), sharey=True, figsize=(10,6))
# fix the dimensions
axes = axes.reshape((len(metrics), len(changed_params)))

for m, ax_row in zip(metrics, axes):
    for c, ax in zip(changed_params, ax_row):
        # group by the other values
        unused_columns = [cn for cn in changed_params if c != cn]
        for group, data in hparams_changed.groupby(unused_columns):
            # plot them with the same line
            ax.plot(data[c], results_means.loc[data.index,m], marker='x', label=str(group))
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
plt.show()
plt.close()
