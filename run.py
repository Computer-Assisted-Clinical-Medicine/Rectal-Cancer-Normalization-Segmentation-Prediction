import logging
import os
from pathlib import Path

import numpy as np

#logger has to be set before tensorflow is imported
tf_logger = logging.getLogger('tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from experiment import Experiment
from SegmentationNetworkBasis.architecture import DVN, CombiNet, UNet, VNet
from tests.create_test_files import create_test_files


#configure loggers
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

tf_logger.setLevel(logging.DEBUG)
#there is too much output otherwise
for h in tf_logger.handlers:
    tf_logger.removeHandler(h)


data_dir = Path('TestData')
experiment_dir = Path('Experiments', 'testExperiment')
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

data_list = np.array([
    str(data_dir/d.stem.split('-')[1]) for d in data_dir.iterdir() if 'label' in d.name
])
#when no files are there, create them
if data_list.size == 0:
    create_test_files(data_dir)
    data_list = np.array([
        str(data_dir/d.stem.split('-')[1]) for d in data_dir.iterdir() if 'label' in d.name
    ])

k_fold = 5

#define the parameters that are constant
init_parameters = {
    "regularize": [True, 'L2', 0.0000001],
    "drop_out": [True, 0.01],
    "activation": "elu",
    "do_batch_normalization": False,
    "do_bias": True,
    "cross_hair": False,
    "do_gradient_clipping" : False,
    "clipping_value" : 50
}

train_parameters = {
    "l_r": 0.001,
    "optimizer": "Adam",
    "epochs" : 10
}

constant_parameters = {
    "init_parameters": init_parameters,
    "train_parameters": train_parameters,
    "loss" : 'DICE'
}

#define the parameters that are being tuned
dimensions = [2, 3]
architectures = [UNet, VNet]

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
tensorboard_command = f'tensorboard --logdir={experiment_dir.absolute()}'
print(f'To see the progress in tensorboard, run:\n{tensorboard_command}')

#generate a set of hyperparameters for each dimension and architecture and run
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

        #add more detailed logger for each network, when problems arise, use debug
        #create file handlers
        fh_info = logging.FileHandler(current_experiment_path/'log_info.txt')
        fh_info.setLevel(logging.INFO)
        fh_info.setFormatter(formatter)
        #add to loggers
        logger.addHandler(fh_info)

        experiment = Experiment(
            hyper_parameters=hyper_parameters,
            name=experiment_name,
            output_path=current_experiment_path,
            folds=k_fold
        )

        experiment.run(data_list)
        experiment.evaluate()

        #remove logger
        logger.removeHandler(fh_info)