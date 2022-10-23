"""Check the normalization to see if there are any large deviations"""
# %%

# pylint:disable=invalid-name, wrong-import-position

import os
from pathlib import Path
from typing import List

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


def read_norm_exp(norm_suffix: str) -> pd.DataFrame:
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
                print(f"\t{loc} not yet finished.")
                continue
            with open(dataset_file, "r", encoding="utf8") as dataset_file:
                dataset = yaml.load(dataset_file, yaml.UnsafeLoader)

            print("Start evaluating the generated images.")
            for pat_name, data in tqdm(dataset.items()):
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

# NORM_SUFFIX = ""
# NORM_SUFFIX = "_tog_idg0.50"
NORM_SUFFIX = "_3_64_0.50_tog_idg0.50"

train_locations = [
    "Frankfurt",
    "Regensburg",
    "Mannheim",
    "all",
    "Not-Frankfurt",
    "Not-Regensburg",
    "Not-Mannheim",
]

# %%

results = read_norm_exp(NORM_SUFFIX)

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

for location in train_locations:

    print(f"Starting with {location}.")

    experiment_group_name = f"Normalization_{location}"

    for channel in range(3):
        norm_exp_dir = (
            exp_group_base_dir
            / experiment_group_name
            / f"Train_Normalization_GAN{NORM_SUFFIX}"
            / f"Train_Normalization_GAN_{channel}{NORM_SUFFIX}"
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
    "generator/disc_latent_loss",
    "generator/disc_real_fake_loss",
    "generator/total_loss",
    "generator/root_mean_squared_error",
    "generator/nmi",
]

for location in train_locations:

    print(f"Starting with {location}.")

    experiment_group_name = f"Normalization_{location}"

    results_list = []
    norm_exp_dir = (
        exp_group_base_dir / experiment_group_name / f"Train_Normalization_GAN{NORM_SUFFIX}"
    )

    train_results_list: List[pd.DataFrame] = []
    for channel in range(3):
        train_file = (
            norm_exp_dir
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
        print("Latent Discriminators")
        plot_disc(train_results, "latent")
        print("Image Discriminators")
        plot_disc(train_results, "image")
