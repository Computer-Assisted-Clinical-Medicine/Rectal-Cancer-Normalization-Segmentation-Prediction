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
import seaborn as sns
import tensorflow as tf
import yaml
from IPython.display import display
from tqdm import tqdm

from networks import auto_encoder
from SegClassRegBasis import evaluation
from utils import plot_disc, plot_disc_exp, plot_with_ma


def read_norm_exp(norm_suffix: str, silent=False) -> pd.DataFrame:
    """Analyze all experiments for one GAN normalization

    Parameters
    ----------
    norm_suffix : str
        The suffix of the normalization

    Returns
    -------
    pd.DataFrame
        The metrics compared to the quantile images
    """

    exp_dir = Path(os.environ["experiment_dir"])

    quantile_location = exp_dir / GROUP_BASE_NAME / "data_preprocessed" / "QUANTILE"
    # read dataset
    quantile_dataset_file = quantile_location / "preprocessing_dataset.yaml"
    with open(quantile_dataset_file, "r", encoding="utf8") as quant_file:
        quantile_dataset = yaml.load(quant_file, yaml.UnsafeLoader)

    results_all_list = []

    for loc in train_locations:
        if not silent:
            print(f"Starting with {loc}.")

        res_list = []
        dataset_file = (
            exp_dir
            / GROUP_BASE_NAME
            / f"Normalization_{loc}"
            / "data_preprocessed"
            / f"GAN_DISCRIMINATORS{norm_suffix}"
            / "preprocessing_dataset.yaml"
        )
        # read dataset
        analysis_file = dataset_file.parent / "analysis.csv"
        if analysis_file.exists():
            results_location = pd.read_csv(analysis_file, sep=";", index_col=0)
        else:
            if not dataset_file.exists():
                if not silent:
                    print(f"\t{loc} not yet finished.")
                continue
            with open(dataset_file, "r", encoding="utf8") as dataset_file:
                dataset = yaml.load(dataset_file, yaml.UnsafeLoader)

            if not silent:
                print("Start evaluating the generated images.")
            for pat_name, data in tqdm(dataset.items(), desc=loc, unit="image"):
                image_gan = experiment_dir / data["image"]
                image_quant = experiment_dir / quantile_dataset[pat_name]["image"]

                for c in range(3):
                    result_metrics = evaluation.evaluate_autoencoder_prediction(
                        str(image_gan), str(image_quant), channel=c
                    )
                    result_metrics["modality"] = c
                    result_metrics["location"] = loc
                    result_metrics["patient_id"] = pat_name

                    res_list.append(result_metrics)

            results_location = pd.DataFrame(res_list)
            results_location.to_csv(analysis_file, sep=";")

        results_all_list.append(results_location)

    if len(results_all_list) > 0:
        return pd.concat(results_all_list).reset_index().drop(columns="index")
    else:
        return None


# %%

data_dir = Path(os.environ["data_dir"])
experiment_dir = Path(os.environ["experiment_dir"])
GROUP_BASE_NAME = "Normalization_Experiment"
exp_group_base_dir = experiment_dir / GROUP_BASE_NAME

train_locations = [
    "Frankfurt",
    "Regensburg",
    "Mannheim",
    "all",
    "Not-Frankfurt",
    "Not-Regensburg",
    "Not-Mannheim",
]

experiments = {
    "def": "",
    "f_64": "_3_64_0.50",
    "f_64_bc": "_3_64_0.50_BetterConv",
    "f_64_bc_lr": "_3_64_0.50_BetterConv_0.00001",
}
suffixes = list(experiments.values())

NORM_SUFFIX = suffixes[-1]

# %%

res_list_suffix = []
for name, s in experiments.items():
    res_suffix = read_norm_exp(s, silent=True)
    if res_suffix is None:
        continue
    res_suffix["normalization_name"] = f"GAN{s}"
    res_suffix["experiment_name"] = name
    res_list_suffix.append(res_suffix)
if len(res_list_suffix) > 0:
    results = pd.concat(res_list_suffix)
else:
    results = None

# %%

if results is not None:
    sns.catplot(
        data=results,
        x="experiment_name",
        y="rmse",
        kind="box",
        row="location",
        col="modality",
    )
    plt.show()
    plt.close()

    sns.catplot(
        data=results,
        x="experiment_name",
        y="structured_similarity_index",
        kind="box",
        row="location",
        col="modality",
    )
    plt.show()
    plt.close()

    sns.catplot(
        data=results,
        x="experiment_name",
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

for location in train_locations:

    print(f"Starting with {location}.")

    experiment_group_name = f"Normalization_{location}"

    for channel in range(3):
        for suffix in suffixes:
            norm_exp_dir = (
                exp_group_base_dir
                / experiment_group_name
                / f"Train_Normalization_GAN{suffix}"
                / f"Train_Normalization_GAN_{channel}{suffix}"
            )

            if not norm_exp_dir.exists():
                continue

            model_file = norm_exp_dir / "generator_with_shapes.png"
            if model_file.exists():
                continue

            with open(norm_exp_dir / "parameters.yaml", "r", encoding="utf8") as f:
                norm_settings = yaml.load(f, yaml.UnsafeLoader)

            print("\tGenerating Model")
            hp_train = norm_settings["hyper_parameters"]["train_parameters"]
            model = auto_encoder(
                inputs=tf.keras.Input(
                    (hp_train["in_plane_dimension"], hp_train["in_plane_dimension"], 1),
                    batch_size=hp_train["batch_size"],
                ),
                **norm_settings["hyper_parameters"]["network_parameters"],
            )
            tf.keras.utils.plot_model(
                model,
                to_file=model_file,
                show_shapes=True,
            )
            print("\tPlotting finished")

# %%

# read the training data

train_results_list: List[pd.DataFrame] = []
for exp_name, suffix in experiments.items():
    for location in train_locations:

        experiment_group_name = f"Normalization_{location}"

        norm_exp_dir = (
            exp_group_base_dir / experiment_group_name / f"Train_Normalization_GAN{suffix}"
        )
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
                train_res["location"] = location
                train_res["experiment"] = exp_name
                train_res["suffix"] = suffix
                train_results_list.append(train_res)
if len(train_results_list) > 0:
    train_results = pd.concat(train_results_list).reset_index()

# %%
# Plot training data

training_metrics = [
    "generator/disc_image_loss",
    "generator/disc_latent_loss",
    "generator/disc_real_fake_loss",
    "generator/total_loss",
    "generator/root_mean_squared_error",
    "generator/nmi",
]

for location in train_locations:
    print(f"Starting with {location}.")
    train_res_loc = train_results.query(
        f"location == '{location}' and suffix == '{NORM_SUFFIX}'"
    )
    if train_res_loc.size == 0:
        continue
    training_metrics_present = [t for t in training_metrics if f"val_{t}" in train_res_loc]
    nrows = int(np.ceil(len(training_metrics_present) / 4))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=4,
        sharex=True,
        sharey=False,
        figsize=(16, nrows * 3.5),
    )
    for metric, ax in zip(training_metrics_present, axes.flat):
        plot_with_ma(ax, train_res_loc, metric)
        ax.set_ylabel(metric)
    for ax in axes.flat[3 : 4 : len(training_metrics_present)]:
        ax.legend()
    for ax in axes.flat[len(training_metrics_present) :]:
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.close()
    print("Latent Discriminators")
    plot_disc(train_res_loc, "latent")
    print("Image Discriminators")
    plot_disc(train_res_loc, "image")

# %%

for location in train_locations:
    print(location)
    plot_disc_exp(
        train_results.query(f"location == '{location}'"),
        list(experiments.keys()),
        [f"{e}-" for e in experiments],
        "image",
    )
