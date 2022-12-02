"""Run all experiments using the experiments file
"""

import os
from pathlib import Path

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


gpu = tf.device(get_gpu(memory_limit=8000))

bot = TelegramBot()
bot.send_message("Starting with training")
bot.send_sticker("CAACAgIAAxkBAAO1Y1evsCpQBMjUMVwxuIp-8GND1B8AAk8AA1m7_CVwHhUv2KjsaioE")

experiments = get_experiments()
num_completed = experiments.completed.sum()
message = f"{num_completed} of {experiments.shape[0]} experiments already finished."
print(message)
bot.send_message(message)

# always load all experiments again, because sometimes experiments are prepared
# during the training
while True:
    experiments = get_experiments()
    all_threads = []
    if np.all(experiments.completed):
        break
    num_completed = experiments.completed.sum()
    for num, (_, exp_pd) in enumerate(experiments[~experiments.completed].iterrows()):
        print(f"Starting with {exp_pd.path}")
        # load experiment
        param_file = experiment_dir / exp_pd.path / "parameters.yaml"
        if not param_file.exists():
            print("Parameter file not found, experiment will be skipped")
            continue
        try:
            exp = Experiment.from_file(param_file)
        except FileNotFoundError:
            print("Files are missing for this experiment")
            continue
        for f in range(exp.folds):
            bot.send_message(
                f"Starting with {exp_pd.path} ({num_completed+num+1}/{experiments.shape[0]}) "
                f"fold {f} ({f+1}/{exp.folds})"
            )
            with gpu:
                try:
                    all_threads.append(run_experiment_fold(exp, f))
                except Exception as exc:  # pylint:disable=broad-except
                    message = f"{exp_pd.path} failed with {exc}"
                    print(message)
                    bot.send_message(message)
            bot.send_message(f"Finished with fold {f}")

        bot.send_message(f"Finished with {exp_pd.path}")

    for thread in all_threads:
        if thread is not None:
            thread.join()

bot.send_message("Finished")
bot.send_sticker()
