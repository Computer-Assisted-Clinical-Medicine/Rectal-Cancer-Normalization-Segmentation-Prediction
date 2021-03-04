import json
import logging
import os
from pathlib import Path

import numpy as np

#logger has to be set before tensorflow is imported
tf_logger = logging.getLogger('tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from experiment import Experiment
from SegmentationNetworkBasis.architecture import DVN, ResNet, UNet, VNet

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

k_fold = 3

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
    "epochs" : 1,
    "early_stopping" : True,
    "reduce_lr_on_plateau" : False
}

constant_parameters = {
    "init_parameters": init_parameters,
    "train_parameters": train_parameters,
    "loss" : 'DICE'
}

#define the parameters that are being tuned
dimensions = [2, 3]
architectures = [UNet, VNet, DVN, ResNet]
# architectures = [VNet]

def generate_folder_name(hyper_parameters):
    epochs = hyper_parameters['train_parameters']['epochs']

    if hyper_parameters['init_parameters']['drop_out'][0]:
        do = 'DO'
    else:
        do = 'nDO'

    if hyper_parameters['init_parameters']['do_batch_normalization']:
        bn = 'BN'
    else:
        bn = 'nBN'

    folder_name = "-".join([
        hyper_parameters['architecture'].get_name() + str(hyper_parameters['dimensions']) + 'D',
        hyper_parameters['loss'],
        do,
        bn,
        str(epochs)
    ])

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
    for a in architectures:
        hyper_parameters = {
            **constant_parameters,
            'dimensions' : d,
            'architecture' : a
        }

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
            preprocessed_dir=preprocessed_dir
        )
        experiments.append(experiment)


# run all folds
for f in range(k_fold):
    for e in experiments:

        #add more detailed logger for each network, when problems arise, use debug
        log_dir = e.output_path / e.fold_dir_names[f]
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        #create file handlers
        fh_info = logging.FileHandler(log_dir/'log_info.txt')
        fh_info.setLevel(logging.INFO)
        fh_info.setFormatter(formatter)
        #add to loggers
        logger.addHandler(fh_info)

        #create file handlers
        fh_debug = logging.FileHandler(log_dir/'log_debug.txt')
        fh_debug.setLevel(logging.DEBUG)
        fh_debug.setFormatter(formatter)
        #add to loggers
        logger.addHandler(fh_debug)

        e.run_fold(f)

        #remove logger
        logger.removeHandler(fh_info)
        logger.removeHandler(fh_debug)

# evaluate all experiments
for e in experiments:
    e.evaluate()