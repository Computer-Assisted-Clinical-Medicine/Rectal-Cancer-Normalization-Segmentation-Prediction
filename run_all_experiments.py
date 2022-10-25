"""Run all experiments using the experiments file
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import telegram

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# pylint: disable=wrong-import-position

import tensorflow as tf

from run_single_experiment import run_experiment_fold
from SegClassRegBasis.experiment import Experiment
from SegClassRegBasis.utils import get_gpu


class TelegramBot:

    """A simple telegram bot, which sends progress messages to the bot with the
    toke "telegram_bot_token" to the chat with the id "telegram_chat_id" found in
    the environmental variables."""

    def __init__(self):
        self.token = os.environ.get("telegram_bot_token", None)
        self.chat_id = os.environ.get("telegram_chat_id", None)
        if self.token is not None and self.chat_id is not None:
            self.bot = telegram.Bot(self.token)
        else:
            print("Set telegram_bot_token and telegram_chat_id to use the telegram bot")
            self.bot = None

    def send_message(self, message: str):
        """Send a message to the phone if variables present, otherwise, do nothing

        Parameters
        ----------
        message : str
            The message
        """
        if self.bot is not None:
            self.bot.send_message(text=message, chat_id=self.chat_id)

    def send_sticker(
        self,
        sticker="CAACAgIAAxkBAAMLY1bguVL3IIg6I5YOMXafXg4ZneEAAkwBAAIw1J0R995vXzeDORwqBA",
    ):
        """Send a sticker to the phone if variables present, otherwise, do nothing

        Parameters
        ----------
        sticker : str, optional
            The id of the sticker, by default a celebratory sticker
        """
        if self.bot is not None:
            self.bot.send_sticker(sticker=sticker, chat_id=self.chat_id)


data_dir = Path(os.environ["data_dir"])
experiment_dir = Path(os.environ["experiment_dir"])
experiments_file = experiment_dir / "Normalization_Experiment" / "experiments.json"
experiments = pd.read_json(experiments_file)

bot = TelegramBot()

gpu = tf.device(get_gpu(memory_limit=2000))
all_threads = []

bot.send_message("Starting with training")
bot.send_sticker("CAACAgIAAxkBAAO1Y1evsCpQBMjUMVwxuIp-8GND1B8AAk8AA1m7_CVwHhUv2KjsaioE")

for num, exp_pd in experiments.iterrows():
    print(f"Starting with {exp_pd.path}")
    exp_path = experiment_dir / exp_pd.path
    if exp_path.exists():
        result_dirs_exist = []
        for task in exp_pd.tasks.values():
            versions = ["best", "final"]
            if task == "segmentation":
                versions += [f"{v}-postprocessed" for v in versions]
            for v in versions:
                result_dirs_exist.append((exp_path / f"results_test_{v}_{task}").exists())
        if np.all(result_dirs_exist):
            print("\tAlready finished with training and evaluated.")
            continue

    # load experiment
    param_file = exp_path / "parameters.yaml"
    assert param_file.exists(), f"Parameter file {param_file} does not exist."
    exp = Experiment.from_file(param_file)

    for f in range(exp.folds):
        bot.send_message(
            f"Starting with {exp_pd.path} ({num+1}/{experiments.shape[1]}) "
            f"fold {f} ({f+1}/{exp.folds})"
        )
        with gpu:
            all_threads.append(run_experiment_fold(exp, f))
        bot.send_message(f"Finished with fold {f}")

    bot.send_message(f"Finished with {exp_pd.path}")

for thread in all_threads:
    thread.join()

bot.send_message("Finished")
bot.send_sticker()
