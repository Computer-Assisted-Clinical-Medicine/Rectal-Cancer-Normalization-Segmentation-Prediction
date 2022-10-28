"""Check the normalization to see if there are any large deviations"""
# %%

# pylint:disable=invalid-name

import os
from pathlib import Path
from typing import List

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_disc_test import preprocessing_parameters
from utils import plot_disc, plot_disc_exp, plot_with_ma

# %%

data_dir = Path(os.environ["data_dir"])
experiment_dir = Path(os.environ["experiment_dir"])
GROUP_BASE_NAME = "Discriminator_Test"
exp_group_base_dir = experiment_dir / GROUP_BASE_NAME

# %%

# get the training data

train_results_list: List[pd.DataFrame] = []
for suffix, _ in preprocessing_parameters:

    results_list: List[pd.DataFrame] = []
    norm_exp_dir = exp_group_base_dir / f"Train_Normalization_GAN{suffix}"

    for c_train_res in range(3):
        train_file = (
            norm_exp_dir
            / f"Train_Normalization_GAN_{c_train_res}{suffix}"
            / "fold-0"
            / "training.csv"
        )
        if train_file.exists():
            train_res = pd.read_csv(train_file, sep=";")
            train_res["channel"] = str(c_train_res)
            train_res["experiment"] = f"disc{suffix}"
            train_results_list.append(train_res)
if len(train_results_list) > 0:
    train_results = pd.concat(train_results_list).reset_index().drop(columns="index")

# %%
# Plot training data

training_metrics = [
    "generator/disc_image_loss",
    "generator/disc_real_fake_loss",
    "generator/total_loss",
    "generator/root_mean_squared_error",
]

for suffix, _ in preprocessing_parameters:
    print(f"Starting with disc{suffix}.")
    exp_data = train_results.query(f"experiment == 'disc{suffix}'")
    if exp_data.size == 0:
        continue
    training_metrics_present = [t for t in training_metrics if f"val_{t}" in exp_data]
    nrows = int(np.ceil(len(training_metrics_present) / 4))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=4,
        sharex=True,
        sharey=False,
        figsize=(16, nrows * 3.5),
    )
    for metric, ax in zip(training_metrics_present, axes.flat):
        plot_with_ma(ax, exp_data, metric)
        ax.set_ylabel(metric)
    for ax in axes.flat[3 : 4 : len(training_metrics_present)]:
        ax.legend()
    for ax in axes.flat[len(training_metrics_present) :]:
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.close()
    print("Image Discriminators")
    plot_disc(exp_data, "image")

print("finished")

# %%

exps: List[str] = []
label_prefixes: List[str] = []
for i in range(2, 7):
    exps.append(f"disc_BetterConv_lr_em{i}")
    label_prefixes.append(f"lr=1e-{i} ch. ")

for i in range(3):
    plot_disc_exp(train_results, exps, label_prefixes, "image", i)
