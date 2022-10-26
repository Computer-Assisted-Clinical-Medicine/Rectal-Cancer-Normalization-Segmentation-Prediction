"""Run all experiments using the experiments file
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

from run_single_experiment import run_experiment_fold
from SegClassRegBasis.experiment import Experiment
from SegClassRegBasis.utils import get_gpu
from utils import TelegramBot

data_dir = Path(os.environ["data_dir"])
experiment_dir = Path(os.environ["experiment_dir"])
experiments_file = experiment_dir / "Normalization_Experiment" / "experiments.json"
experiments = pd.read_json(experiments_file)
experiments["completed"] = False

bot = TelegramBot()

gpu = tf.device(get_gpu(memory_limit=2000))
all_threads = []

bot.send_message("Starting with training")
bot.send_sticker("CAACAgIAAxkBAAO1Y1evsCpQBMjUMVwxuIp-8GND1B8AAk8AA1m7_CVwHhUv2KjsaioE")

for exp_id, exp_pd in experiments.iterrows():
    exp_path = experiment_dir / exp_pd.path
    if exp_path.exists():
        result_dirs_exist = []
        for task in exp_pd.tasks.values():
            versions = exp_pd.versions
            if task == "segmentation":
                versions += [f"{v}-postprocessed" for v in versions]
            for v in versions:
                names = ["test"]
                if exp_pd.external:
                    names.append("external_testset")
                for n in names:
                    result_dirs_exist.append(
                        (exp_path / f"results_{n}_{v}_{task}").exists()
                    )
        if np.all(result_dirs_exist):
            print(f"{exp_pd.path} already finished with training and evaluated.")
            experiments.loc[exp_id, "completed"] = True

num_completed = experiments.completed.sum()
message = f"{num_completed} of {experiments.shape[0]} experiments already finished."
print(message)
bot.send_message(message)

for num, (_, exp_pd) in enumerate(experiments[~experiments.completed].iterrows()):
    print(f"Starting with {exp_pd.path}")
    # load experiment
    param_file = experiment_dir / exp_pd.path / "parameters.yaml"
    assert param_file.exists(), f"Parameter file {param_file} does not exist."
    exp = Experiment.from_file(param_file)

    for f in range(exp.folds):
        bot.send_message(
            f"Starting with {exp_pd.path} ({num+num_completed+1}/{experiments.shape[0]}) "
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
    thread.join()

bot.send_message("Finished")
bot.send_sticker()
