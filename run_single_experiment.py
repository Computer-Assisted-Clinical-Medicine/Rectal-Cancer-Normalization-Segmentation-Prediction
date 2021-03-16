import argparse
import logging
import os
from pathlib import Path

#logger has to be set before tensorflow is imported
tf_logger = logging.getLogger('tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from experiment import Experiment
from run import plot_hparam_comparison

def init_argparse():
    parser = argparse.ArgumentParser(
        description='Do the training of one single fold.'
    )
    parser.add_argument(
        '-f', '--fold',
        metavar='fold',
        type=int,
        nargs='?',
        help='The number of the folds to process.'
    )
    parser.add_argument(
        '-e', '--experiment_dir',
        metavar='experiment_dir',
        type=str,
        nargs='?',
        help='The directory were the experiment os located at with the parameters.yaml file.'
    )
    return parser

parser = init_argparse()
args = parser.parse_args()

#configure loggers
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

tf_logger.setLevel(logging.DEBUG)
#there is too much output otherwise
for h in tf_logger.handlers:
    tf_logger.removeHandler(h)


data_dir = Path(os.environ['data_dir'])
experiment_dir = Path(os.environ['experiment_dir'])

current_experiment_dir = Path(args.experiment_dir)
f = args.fold

# load experiment
param_file = current_experiment_dir / 'parameters.yaml'
assert param_file.exists(), f'Parameter file {param_file} does not exist.'
experiment = Experiment.from_file(param_file)

assert f < experiment.folds, f'Fold number {f} is higher than the maximim number {experiment.folds}.'

#add more detailed logger for each network, when problems arise, use debug
log_dir = experiment.output_path / experiment.fold_dir_names[f]
if not log_dir.exists():
    log_dir.mkdir(parents=True)

# initialize loggers
# create formatter
formatter = logging.Formatter('%(levelname)s: %(name)s - %(funcName)s (l.%(lineno)d): %(message)s')

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

# run experiment
experiment.run_fold(f)
# try to evaluate it (this will only work if this is the last fold)
try:
    experiment.evaluate()
except:
    print('Could not evaluate the experiment (happens if not all folds are finished).')
else:
    print('Evaluation finished.')

try:
    plot_hparam_comparison(experiment_dir)
except Exception as e:
    print(e)
    print('Plotting of hyperparameter comparison failed.')
else:
    print('Hyperparameter comparison was plotted.')

#remove logger
logger.removeHandler(fh_info)
logger.removeHandler(fh_debug)