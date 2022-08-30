"""
Run Aa single experiment (this is only used when running on the cluster)
"""
import argparse
import logging
import os
from pathlib import Path

import filelock

# logger has to be set before tensorflow is imported
tf_logger = logging.getLogger("tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# pylint: disable=wrong-import-position

from SegClassRegBasis.experiment import Experiment
from SegClassRegBasis.utils import configure_logging


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


def run_experiment_fold(experiment: Experiment, fold: int):
    """Run the fold of a single experiment, this function mainly handles the
    logging and then calls experiment.run_fold(fold)

    Parameters
    ----------
    experiment : Experiment
        The current experiment
    fold : int
        The Number of the fold
    """

    try:
        experiment.run_fold(fold)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception(str(exc))
        raise exc


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

assert f < exp.folds, f"Fold number {f} is higher than the maximum number {exp.folds}."

# add more detailed logger for each network, when problems arise, use debug
log_dir = exp.output_path / exp.fold_dir_names[f]
if not log_dir.exists():
    log_dir.mkdir(parents=True)

# only have one process running for each fold
with filelock.FileLock(log_dir / "lock_fold.txt.lock", timeout=1):
    log_formatter = logging.Formatter(
        "%(levelname)s: %(name)s - %(funcName)s (l.%(lineno)d): %(message)s"
    )

    # create file handlers
    fh_info = logging.FileHandler(log_dir / "log_info.txt")
    fh_info.setLevel(logging.INFO)
    fh_info.setFormatter(log_formatter)
    # add to loggers
    logger.addHandler(fh_info)

    # create file handlers
    fh_debug = logging.FileHandler(log_dir / "log_debug.txt")
    fh_debug.setLevel(logging.DEBUG)
    fh_debug.setFormatter(log_formatter)
    # add to loggers
    logger.addHandler(fh_debug)

    run_experiment_fold(exp, f)

    # try to evaluate it (this will only work if this is the last fold)
    try:
        exp.evaluate()
    except FileNotFoundError:
        print("Could not evaluate the experiment (happens if not all folds are finished).")
    else:
        print("Evaluation finished.")

    # try to evaluate the external testset
    if exp.external_test_set is not None:
        try:
            exp.evaluate_external_testset()
        except FileNotFoundError:
            print(
                "Could not evaluate the experiment (happens if not all folds are finished)."
            )
        else:
            print("Evaluation finished.")
