"""
Run Aa single experiment (this is only used when running on the cluster)
"""
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
from run_single_experiment import init_argparse, configure_loggers


def evaluate_experiment_fold(experiment: Experiment, fold: int):
    """Run the fold of a single experiment, this function mainly handles the
    logging and then calls experiment.evaluate_fold_complete(fold)

    Parameters
    ----------
    experiment : Experiment
        The current experiment
    fold : int
        The Number of the fold
    """

    try:
        experiment.evaluate_fold(fold)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception(str(exc))
        raise exc


if __name__ == "__main__":

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
    lock_file = log_dir / "lock_fold.txt.lock"
    configure_loggers(logger, log_dir, "_eval")
    with filelock.FileLock(lock_file, timeout=1):
        evaluate_experiment_fold(exp, f)

        # try to evaluate it (this will only work if this is the last fold)
        try:
            exp.evaluate()
        except FileNotFoundError:
            print(
                "Could not evaluate the experiment (happens if not all folds are finished)."
            )
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

    if lock_file.exists():
        lock_file.unlink()
