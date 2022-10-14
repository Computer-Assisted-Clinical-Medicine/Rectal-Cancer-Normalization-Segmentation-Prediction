"""Check the normalization to see if there are any large deviations"""
# %%

# pylint:disable=invalid-name

import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml
from IPython.display import display
from tqdm.autonotebook import tqdm

from networks import auto_encoder
from SegClassRegBasis import evaluation

# %%

data_dir = Path(os.environ["data_dir"])
experiment_dir = Path(os.environ["experiment_dir"])
GROUP_BASE_NAME = "Normalization_Experiment"
exp_group_base_dir = experiment_dir / GROUP_BASE_NAME

# NORM_SUFFIX = ""
NORM_SUFFIX = "_tog"
# NORM_SUFFIX = "_4_64_0.50"
# NORM_SUFFIX = "_0_64_0.50"
# NORM_SUFFIX = "_4_64_0.50_tog"

quantile_location = exp_group_base_dir / "data_preprocessed" / "QUANTILE"
# read dataset
quantile_dataset_file = quantile_location / "preprocessing_dataset.yaml"
with open(quantile_dataset_file, "r", encoding="utf8") as f:
    quantile_dataset = yaml.load(f, yaml.UnsafeLoader)

# %%

results_all_list = []

for location in [
    "all",
    "Frankfurt",
    "Regensburg",
    "Mannheim",
    "Not-Frankfurt",
    "Not-Regensburg",
    "Not-Mannheim",
]:

    print(f"Starting with {location}.")

    experiment_group_name = f"Normalization_{location}"

    results_list = []
    dataset_file = (
        exp_group_base_dir
        / experiment_group_name
        / "data_preprocessed"
        / f"GAN_DISCRIMINATORS{NORM_SUFFIX}"
        / "preprocessing_dataset.yaml"
    )
    # read dataset
    analysis_file = dataset_file.parent / "analysis.csv"
    if analysis_file.exists():
        results_location = pd.read_csv(analysis_file, sep=";", index_col=0)
    else:
        if not dataset_file.exists():
            print(f"\t{location} not yet finished.")
            continue
        with open(dataset_file, "r", encoding="utf8") as f:
            dataset = yaml.load(f, yaml.UnsafeLoader)

        print("Start evaluating the generated images.")
        for pat_name, data in tqdm(dataset.items()):
            image_gan = experiment_dir / data["image"]
            image_quant = experiment_dir / quantile_dataset[pat_name]["image"]

            for channel in range(3):
                result_metrics = evaluation.evaluate_autoencoder_prediction(
                    str(image_gan), str(image_quant), channel=channel
                )
                result_metrics["modality"] = channel
                result_metrics["location"] = location
                result_metrics["patient_id"] = pat_name

                results_list.append(result_metrics)

        results_location = pd.DataFrame(results_list)
        results_location.to_csv(analysis_file, sep=";")

    results_all_list.append(results_location)

if len(results_all_list) > 0:
    results = pd.concat(results_all_list).reset_index().drop(columns="index")
else:
    results = None

# %%

if results is not None:
    sns.catplot(data=results, y="rmse", kind="box", row="location", col="modality")
    plt.show()
    plt.close()

    sns.catplot(
        data=results,
        y="structured_similarity_index",
        kind="box",
        row="location",
        col="modality",
    )
    plt.show()
    plt.close()

    sns.catplot(
        data=results,
        y="norm_mutual_inf",
        kind="box",
        row="location",
        col="modality",
    )
    plt.show()
    plt.close()

# %%

if results is not None:
    mean_results = results.groupby(["patient_id", "modality"]).median()

    display(
        mean_results.sort_values("structured_similarity_index")[
            ["rmse", "norm_mutual_inf", "structured_similarity_index"]
        ]
    )

    display(
        results.sort_values("structured_similarity_index")[
            [
                "patient_id",
                "location",
                "rmse",
                "norm_mutual_inf",
                "structured_similarity_index",
            ]
        ]
    )

# %%
# plot the generator


for location in [
    "all",
    "Frankfurt",
    "Regensburg",
    "Mannheim",
    "Not-Frankfurt",
    "Not-Regensburg",
    "Not-Mannheim",
]:

    print(f"Starting with {location}.")

    experiment_group_name = f"Normalization_{location}"

    for channel in range(3):
        exp_dir = (
            exp_group_base_dir
            / experiment_group_name
            / f"Train_Normalization_GAN{NORM_SUFFIX}"
            / f"Train_Normalization_GAN_{channel}{NORM_SUFFIX}"
        )

        if not exp_dir.exists():
            continue

        model_file = exp_dir / "generator_with_shapes.png"
        if model_file.exists():
            continue

        with open(exp_dir / "parameters.yaml", "r", encoding="utf8") as f:
            norm_settings = yaml.load(f, yaml.UnsafeLoader)

        hp_train = norm_settings["hyper_parameters"]["train_parameters"]
        hp_net = norm_settings["hyper_parameters"]["network_parameters"]
        model = auto_encoder(
            inputs=tf.keras.Input(
                (hp_train["in_plane_dimension"], hp_train["in_plane_dimension"], 1),
                batch_size=hp_train["batch_size"],
            ),
            depth=hp_net["depth"],
            filter_base=hp_net["filter_base"],
            skip_edges=hp_net["skip_edges"],
            output_min=hp_net.get("output_min", None),
            output_max=hp_net.get("output_max", None),
            variational=hp_net.get("variational", False),
        )
        tf.keras.utils.plot_model(
            model,
            to_file=model_file,
            show_shapes=True,
        )

# %%
# Plot training data


def plot_disc(res: pd.DataFrame, disc_type: str):
    """Plot the image or latent discriminators"""
    disc = [
        c[6 + len(disc_type) : -5]
        for c in res.columns
        if c.startswith(f"disc_{disc_type}/") and c.endswith("loss") and len(c) > 17
    ]
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
        for val in ("", "val_"):
            fields = [
                f"{val}disc_{disc_type}/{disc}/loss",
                f"{val}disc_{disc_type}/{disc}_{disc_metric}",
                f"{val}generator-{disc_type}/{disc}/loss",
                f"{val}generator-{disc_type}/{disc}_{disc_metric}",
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
                        f"{val}disc_image_gen/{disc}/loss",
                        f"{val}disc_image_gen/{disc}_{disc_metric}",
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
            for c in range(3):
                res_channel = res.query(f"channel == '{c}'")
                for num, (field, name) in enumerate(
                    zip(
                        fields,
                        names,
                    )
                ):
                    if field in res_channel.columns and res_channel.size > 0:
                        plot_with_ma(
                            axes_line[num],
                            res_channel.epoch,
                            res_channel[field],
                            f"{val}{c}",
                        )
                        axes_line[num].set_ylabel(name)
                        last_present = max(last_present, num)
        axes_line[last_present].legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_with_ma(ax_ma, x, y, label, N=10):
    """Plot a line with the value in the background and the moving average on top"""
    p = ax_ma.plot(
        x,
        y,
        alpha=0.2,
    )
    # plot with moving average
    ax_ma.plot(
        x,
        np.convolve(
            np.pad(y, (N // 2 - 1, N // 2), mode="reflect"), np.ones(N) / N, mode="valid"
        ),
        label=label,
        color=p[-1].get_color(),
    )


training_metrics = [
    "val_generator/disc_image_loss",
    "val_generator/disc_latent_loss",
    "val_generator/disc_real_fake_loss",
    "val_generator/total_loss",
    "val_generator/root_mean_squared_error",
    "val_generator/nmi",
]

for location in [
    "all",
    "Frankfurt",
    "Regensburg",
    "Mannheim",
    "Not-Frankfurt",
    "Not-Regensburg",
    "Not-Mannheim",
]:

    print(f"Starting with {location}.")

    experiment_group_name = f"Normalization_{location}"

    results_list = []
    exp_dir = (
        exp_group_base_dir / experiment_group_name / f"Train_Normalization_GAN{NORM_SUFFIX}"
    )

    train_results_list: List[pd.DataFrame] = []
    for channel in range(3):
        train_file = (
            exp_dir
            / f"Train_Normalization_GAN_{channel}{NORM_SUFFIX}"
            / "fold-0"
            / "training.csv"
        )
        if train_file.exists():
            train_res = pd.read_csv(train_file, sep=";")
            train_res["channel"] = str(channel)
            train_results_list.append(train_res)
    if len(train_results_list) > 0:
        train_results = pd.concat(train_results_list).reset_index()
        training_metrics_present = [t for t in training_metrics if t in train_results]
        nrows = int(np.ceil(len(training_metrics_present) / 4))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=4,
            sharex=True,
            sharey=False,
            figsize=(16, nrows * 3.5),
        )
        for metric, ax in zip(training_metrics_present, axes.flat):
            sns.lineplot(
                data=train_results,
                x="epoch",
                y=metric,
                hue="channel",
                ax=ax,
            )
        for ax in axes.flat[len(training_metrics_present) :]:
            ax.set_axis_off()
        plt.tight_layout()
        plt.show()
        plt.close()
        print("Latent Discriminators")
        plot_disc(train_results, "latent")
        print("Image Discriminators")
        plot_disc(train_results, "image")
