import logging
from pathlib import Path

import numpy as np

from experiment import Experiment
from SegmentationNetworkBasis.architecture import DVN, CombiNet, UNet, VNet


#configure logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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

#print errors
ch = logging.StreamHandler()
ch.setLevel(level=logging.ERROR)
ch.setFormatter(formatter)
logger.addHandler(ch)

data_list = np.array([str(d) for d in data_dir.iterdir() if d.is_dir()])

k_fold = 1 #TODO: increase
dimensions_and_architectures = ([2, UNet], [2, VNet], [3, UNet], [3, VNet])

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
    "epochs" : 1 #TODO: increase
}

constant_parameters = {
    "init_parameters": init_parameters,
    "train_parameters": train_parameters,
    "loss" : 'DICE'
}

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

hyper_parameters = []
names = []

#generate a set of hyperparameters for each dimension and architecture
for d, a in dimensions_and_architectures:
    params = {
        **constant_parameters,
        'dimensions' : d,
        'architecture' : a
    }
    hyper_parameters.append(params)

    #define experiment
    names.append(generate_folder_name(params))

#run the experiments
for params, experiment_name in zip(hyper_parameters, names):
    
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
        hyper_parameters=params,
        name=experiment_name,
        output_path=current_experiment_path
    )

    experiment.run(data_list, k_fold)
    experiment.evaluate()

    #remove logger
    logger.removeHandler(fh_info)

#TODO: check input (size, orientation, label classes correct)
#TODO: Make it more clear where the logs are saved
#TODO: update config or get rid of it completely
#TODO: Make hyperparameters work with tensorbaord
#TODO: Make plots nicer
#TODO: take project-specific filenames out of the submodule
