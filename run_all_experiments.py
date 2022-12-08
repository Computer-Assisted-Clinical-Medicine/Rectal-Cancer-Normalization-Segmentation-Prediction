"""Run all experiments using the experiments file
"""

import os
from pathlib import Path
from threading import Thread
from typing import List

import numpy as np
import pandas as pd

import tensorflow as tf

from run_single_experiment import run_experiment_fold
from SegClassRegBasis.experiment import Experiment
from SegClassRegBasis.utils import get_gpu
from utils import TelegramBot

experiment_dir = Path(os.environ["experiment_dir"])


def get_experiments():
    """Get all experiments and see which are completed"""
    experiments_file = experiment_dir / "Normalization_Experiment" / "experiments.json"
    exp_df = pd.read_json(experiments_file)
    exp_df["completed"] = False

    for exp_id, exp_row in exp_df.iterrows():
        exp_path = experiment_dir / exp_row.path
        if exp_path.exists():
            result_files_exist = []
            for task in exp_row.tasks.values():
                versions = exp_row.versions
                if task == "segmentation":
                    versions += [f"{ver}-postprocessed" for ver in versions]
                for ver in versions:
                    names = ["test"]
                    if exp_row.external:
                        names.append("external_testset")
                    for exp_name in names:
                        res_file = (
                            exp_path
                            / f"results_{exp_name}_{ver}_{task}"
                            / "evaluation-all-files.h5"
                        )
                        result_files_exist.append(res_file.exists())
            if np.all(result_files_exist):
                print(f"{exp_row.path} already finished with training and evaluated.")
                exp_df.loc[exp_id, "completed"] = True
    return exp_df


def run_all_experiments(exp_dir: Path, gpu, bot: TelegramBot) -> bool:
    """Run all experiments in the experiment dir with the gpu and send out messages
    to the bot.

    Parameters
    ----------
    exp_dir : Path
        The experiment directory
    gpu : Tensorflow Device
        The device to use (used as context manager)
    bot : TelegramBot
        The bot to send out messages with

    Returns
    -------
    bool
        The return values signifies if all experiments are completed or not
    """

    # always load all experiments again, because sometimes experiments are prepared
    # during the training
    experiments = get_experiments()
    all_threads = []
    if np.all(experiments.completed):
        return True
    for _, exp_pd in experiments[~experiments.completed].iterrows():
        if "UNet" in exp_pd.path:
            continue
        # if exp_pd.dimensions == 2:
        #     continue
        print(f"Starting with {exp_pd.path}")
        # load experiment
        param_file = exp_dir / exp_pd.path / "parameters.yaml"
        if not param_file.exists():
            print("Parameter file not found, experiment will be skipped")
            continue
        try:
            exp = Experiment.from_file(param_file)
        except FileNotFoundError:
            print("Files are missing for this experiment")
            continue
        all_threads += run_experiment(gpu, bot, experiments, exp)

        bot.send_message(f"Finished with {exp_pd.path}")

    for thread in all_threads:
        if thread is not None:
            thread.join()

    return False


def run_experiment(
    gpu, bot: TelegramBot, experiments: pd.DataFrame, exp: Experiment
) -> List[Thread]:
    """Run a single experiment with all folds

    Parameters
    ----------
    gpu : Tensorflow Device
        The device to use (used as context manager)
    bot : TelegramBot
        The bot to send out messages with
    experiments : pd.DataFrame
        The dataframe with all experiments. This is used to calculate the number
        of the current and remaining experiments.
    exp : Experiment
        The experiment object

    Returns
    -------
    List[Thread]
        Returns a list of all threads doing evaluation, they should be joined before
        doing more things.
    """
    n_comp = experiments.completed.sum()
    n_exp = experiments.shape[0]
    threads = []
    for f in range(exp.folds):
        bot.send_message(
            f"Starting with {exp.output_path_rel} ({n_comp+1}/{n_exp}) "
            f"fold {f} ({f+1}/{exp.folds})"
        )
        with gpu:
            try:
                threads.append(run_experiment_fold(exp, f))
            except Exception as exc:  # pylint:disable=broad-except
                message = f"{exp.output_path_rel} failed with {type(exc).__name__}: "
                error_message = str(exc)
                full_msg = message + error_message
                if len(error_message) > 400:
                    error_message = error_message[:200] + "\n...\n" + error_message[-200:]
                message += error_message
                print(full_msg)
                bot.send_message(message)
        bot.send_message(f"Finished with fold {f}")
    return threads


if __name__ == "__main__":

    tf_device = tf.device(get_gpu(memory_limit=8000))

    telegram_bot = TelegramBot()
    telegram_bot.send_message("Starting with training")
    telegram_bot.send_sticker(
        "CAACAgIAAxkBAAO1Y1evsCpQBMjUMVwxuIp-8GND1B8AAk8AA1m7_CVwHhUv2KjsaioE"
    )

    FINISHED = False
    while not FINISHED:
        FINISHED = run_all_experiments(experiment_dir, tf_device, telegram_bot)

    telegram_bot.send_message("Finished")
    telegram_bot.send_sticker()
