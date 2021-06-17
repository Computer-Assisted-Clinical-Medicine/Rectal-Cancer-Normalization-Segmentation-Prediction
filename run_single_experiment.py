"""
Run Aa single experiment (this is only used when running on the cluster)
"""
import argparse
import logging
import os
from pathlib import Path

# logger has to be set before tensorflow is imported
tf_logger = logging.getLogger("tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# pylint: disable=wrong-import-position

import tensorflow as tf

from experiment import Experiment
from utils import configure_logging, plot_hparam_comparison

def init_argparse():
    """
    initialize the parser
    """
    argpar = argparse.ArgumentParser(description="Do the training of one single fold.")
    argpar.add_argument(
        "-f",
        "--fold",
        metavar="fold",
        type=int,
        nargs="?",
        help="The number of the folds to process.",
    )
    argpar.add_argument(
        "-e",
        "--experiment_dir",
        metavar="experiment_dir",
        type=str,
        nargs="?",
        help="The directory were the experiment os located at with the parameters.yaml file.",
    )
    return argpar


def run_experiment_fold(experiment: Experiment, fold: int, base_logger: logging.Logger):
    """Run the fold of a singel experiment, this function mainly handles the
    logging and then calls experiment.run_fold(fold)

    Parameters
    ----------
    experiment : Experiment
        The current experiment
    fold : int
        The Number of the fold
    base_logger : logging.Logger
        The logger to use
    """
    # add more detailed logger for each network, when problems arise, use debug
    fold_dir = experiment.output_path / experiment.fold_dir_names[fold]
    if not fold_dir.exists():
        fold_dir.mkdir(parents=True)

    log_formatter = logging.Formatter(
        "%(levelname)s: %(name)s - %(funcName)s (l.%(lineno)d): %(message)s"
    )

    # create file handlers
    fh_info = logging.FileHandler(fold_dir / "log_info.txt")
    fh_info.setLevel(logging.INFO)
    fh_info.setFormatter(log_formatter)
    # add to loggers
    base_logger.addHandler(fh_info)

    # create file handlers
    fh_debug = logging.FileHandler(fold_dir / "log_debug.txt")
    fh_debug.setLevel(logging.DEBUG)
    fh_debug.setFormatter(log_formatter)
    # add to loggers
    base_logger.addHandler(fh_debug)

    try:
        experiment.run_fold(fold)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception(str(exc))
        raise exc

    # remove logger
    base_logger.removeHandler(fh_info)
    base_logger.removeHandler(fh_debug)


parser = init_argparse()
args = parser.parse_args()

# configure loggers
logger = configure_logging(tf_logger)


data_dir = Path(os.environ["data_dir"])
experiment_dir = Path(os.environ["experiment_dir"])

current_experiment_dir = Path(args.experiment_dir)
f = args.fold

# load experiment
param_file = current_experiment_dir / "parameters.yaml"
assert param_file.exists(), f"Parameter file {param_file} does not exist."
exp = Experiment.from_file(param_file)

assert f < exp.folds, f"Fold number {f} is higher than the maximim number {exp.folds}."

# add more detailed logger for each network, when problems arise, use debug
log_dir = exp.output_path / exp.fold_dir_names[f]
if not log_dir.exists():
    log_dir.mkdir(parents=True)

with tf.device("/device:GPU:0"):
    run_experiment_fold(exp, f, logger)

# try to evaluate it (this will only work if this is the last fold)
try:
    exp.evaluate()
except FileNotFoundError:
    print("Could not evaluate the experiment (happens if not all folds are finished).")
else:
    print("Evaluation finished.")

# try to evaluate the external testset
try:
    exp.evaluate_external_testset()
except FileNotFoundError:
    print("Could not evaluate the experiment (happens if not all folds are finished).")
else:
    print("Evaluation finished.")

try:
    plot_hparam_comparison(experiment_dir)
    plot_hparam_comparison(experiment_dir, postprocessed=True)
except FileNotFoundError:
    print("Plotting of hyperparameter comparison failed.")
else:
    print("Hyperparameter comparison was plotted.")

try:
    plot_hparam_comparison(experiment_dir, external=True)
    plot_hparam_comparison(experiment_dir, external=True, postprocessed=True)
except FileNotFoundError:
    print("Plotting of hyperparameter comparison on the external testset failed.")
else:
    print("Hyperparameter comparison on the external testset was plotted.")
