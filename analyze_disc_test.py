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


# %%

data_dir = Path(os.environ["data_dir"])
experiment_dir = Path(os.environ["experiment_dir"])
GROUP_BASE_NAME = "Discriminator_Test"
exp_group_base_dir = experiment_dir / GROUP_BASE_NAME

# %%
# Plot training data


def plot_disc(res: pd.DataFrame, disc_type: str):
    """Plot the image or latent discriminators"""
    disc = [
        c[6 + len(disc_type) : -5]
        for c in res.columns
        if c.startswith(f"disc_{disc_type}/") and c.endswith("loss") and len(c) > 17
    ]
    if len(disc) == 0:
        return
    img_gen_list = [c for c in res.columns if c.startswith("disc_image_gen")]
    image_gen_pres = len(img_gen_list) > 0 and disc_type == "image"
    if image_gen_pres:
        ncols = 6
    else:
        ncols = 4
    _, axes_disc = plt.subplots(
        nrows=len(disc),
        ncols=ncols,
        sharex=True,
        sharey=False,
        figsize=(4 * ncols, len(disc) * 4),
    )
    for disc, axes_line in zip(disc, axes_disc):
        disc_start = f"disc_{disc_type}/{disc}_"
        disc_metric = [c for c in res.columns if c.startswith(disc_start)][0].partition(
            disc_start
        )[-1]
        if disc_metric == "RootMeanSquaredError":
            disc_metric_name = "RMSE"
        else:
            disc_metric_name = disc_metric
        last_present = 0
        fields = [
            f"disc_{disc_type}/{disc}/loss",
            f"disc_{disc_type}/{disc}_{disc_metric}",
            f"generator-{disc_type}/{disc}/loss",
            f"generator-{disc_type}/{disc}_{disc_metric}",
        ]
        names = [
            f"Disc-{disc_type.capitalize()} {disc} loss",
            f"Disc-{disc_type.capitalize()} {disc} {disc_metric_name}",
            f"Generator {disc} loss",
            f"Generator {disc} {disc_metric_name}",
        ]
        if image_gen_pres:
            fields = (
                fields[:2]
                + [
                    f"disc_image_gen/{disc}/loss",
                    f"disc_image_gen/{disc}_{disc_metric}",
                ]
                + fields[2:]
            )
            names = (
                names[:2]
                + [
                    f"Disc-Image-Gen {disc} loss",
                    f"Disc-Image-Gen {disc} {disc_metric_name}",
                ]
                + names[2:]
            )
        for num, (field, name) in enumerate(
            zip(
                fields,
                names,
            )
        ):
            if field in res.columns:
                plot_with_ma(axes_line[num], res, field)
                axes_line[num].set_ylabel(name)
                last_present = max(last_present, num)
        axes_line[last_present].legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_with_ma(ax_ma, df, field, N=10):
    """Plot a line with the value in the background and the moving average on top"""
    for val in ("", "val_"):
        for c in range(3):
            df_channel = df.query(f"channel == '{c}'")
            if df_channel.size == 0:
                continue
            p = ax_ma.plot(
                df_channel.epoch,
                df_channel[val + field],
                alpha=0.4,
            )
            # plot with moving average
            ax_ma.plot(
                df_channel.epoch,
                np.convolve(
                    np.pad(df_channel[val + field], (N // 2 - 1, N // 2), mode="edge"),
                    np.ones(N) / N,
                    mode="valid",
                ),
                label=f"{val}{c}",
                color=p[-1].get_color(),
            )


training_metrics = [
    "generator/disc_image_loss",
    "generator/disc_real_fake_loss",
    "generator/total_loss",
    "generator/root_mean_squared_error",
]

for suffix, _ in preprocessing_parameters:

    print(f"Starting with GAN{suffix}.")

    results_list: List[pd.DataFrame] = []
    norm_exp_dir = exp_group_base_dir / f"Train_Normalization_GAN{suffix}"

    train_results_list: List[pd.DataFrame] = []
    for channel in range(3):
        train_file = (
            norm_exp_dir
            / f"Train_Normalization_GAN_{channel}{suffix}"
            / "fold-0"
            / "training.csv"
        )
        if train_file.exists():
            train_res = pd.read_csv(train_file, sep=";")
            train_res["channel"] = str(channel)
            train_results_list.append(train_res)
    if len(train_results_list) > 0:
        train_results = pd.concat(train_results_list).reset_index()
        training_metrics_present = [
            t for t in training_metrics if f"val_{t}" in train_results
        ]
        nrows = int(np.ceil(len(training_metrics_present) / 4))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=4,
            sharex=True,
            sharey=False,
            figsize=(16, nrows * 3.5),
        )
        for metric, ax in zip(training_metrics_present, axes.flat):
            plot_with_ma(ax, train_results, metric)
            ax.set_ylabel(metric)
        for ax in axes.flat[3 : 4 : len(training_metrics_present)]:
            ax.legend()
        for ax in axes.flat[len(training_metrics_present) :]:
            ax.set_axis_off()
        plt.tight_layout()
        plt.show()
        plt.close()
        print("Image Discriminators")
        plot_disc(train_results, "image")
