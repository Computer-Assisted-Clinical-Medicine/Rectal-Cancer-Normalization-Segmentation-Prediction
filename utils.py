"""
Miscellaneous functions
"""

import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
import telegram
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm

from SegClassRegBasis.architecture import DeepLabv3plus, DenseTiramisu, UNet


def generate_folder_name(parameters):
    """
    Make a name summarizing the hyperparameters.
    """
    epochs = parameters["train_parameters"]["epochs"]

    params = [
        parameters["architecture"].get_name() + str(parameters["dimensions"]) + "D",
        parameters["loss"],
    ]

    # TODO: move this logic into the network
    if parameters["architecture"] is UNet:
        # attention parameters
        if "encoder_attention" in parameters["network_parameters"]:
            if parameters["network_parameters"]["encoder_attention"] is not None:
                params.append(parameters["network_parameters"]["encoder_attention"])
        if "attention" in parameters["network_parameters"]:
            if parameters["network_parameters"]["attention"]:
                params.append("Attn")

        # residual connections if it is an attribute
        if "res_connect" in parameters["network_parameters"]:
            if parameters["network_parameters"]["res_connect"]:
                params.append("Res")
            else:
                params.append("nRes")

        # filter multiplier
        params.append("f_" + str(parameters["network_parameters"]["n_filters"][0] // 8))

        # batch norm
        if parameters["network_parameters"]["do_batch_normalization"]:
            params.append("BN")
        else:
            params.append("nBN")

        # dropout
        if parameters["network_parameters"]["drop_out"][0]:
            params.append("DO")
        else:
            params.append("nDO")
    elif parameters["architecture"] is DenseTiramisu:
        params.append("gr_" + str(parameters["network_parameters"]["growth_rate"]))

        params.append(
            "nl_" + str(len(parameters["network_parameters"]["layers_per_block"]))
        )
    elif parameters["architecture"] is DeepLabv3plus:
        params.append(str(parameters["network_parameters"]["backbone"]))

        params.append(
            "aspp_"
            + "_".join([str(n) for n in parameters["network_parameters"]["aspp_rates"]])
        )
    else:
        raise NotImplementedError(f'{parameters["architecture"]} not implemented')

    # normalization
    norm_name = parameters["preprocessing_parameters"]["normalizing_method"].name
    if norm_name == "GAN_DISCRIMINATORS":
        gan_suffix = get_gan_suffix(parameters)
        norm_name += gan_suffix

    params.append(str(norm_name))

    # object fraction
    params.append(
        f'obj_{int(parameters["train_parameters"]["percent_of_object_samples"]*100):03d}%'
    )

    # add epoch number
    params.append(str(epochs))

    folder_name = "-".join(params)

    return folder_name


def get_gan_suffix(parameters: Dict) -> str:
    """Take the hyperparameters and get the gan suffix

    Parameters
    ----------
    parameters : Dict
        The hyperparameters

    Returns
    -------
    str
        The suffix
    """
    norm_params = parameters["preprocessing_parameters"]["normalization_parameters"]
    depth = norm_params["depth"]
    f_base = norm_params["filter_base"]
    sigma = norm_params["smoothing_sigma"]
    image_weight = norm_params.get("image_weight", 1)
    image_gen_weight = norm_params.get("image_gen_weight", 1)
    disc_n_conv = norm_params.get("disc_n_conv", 3)
    disc_filter_base = norm_params.get("disc_filter_base", 32)
    disc_type = norm_params.get("disc_type", "SimpleConv")
    disc_start_lr = norm_params.get("disc_start_lr", 0.05)
    if depth == 3 and f_base == 16 and np.isclose(sigma, 1):
        gan_suffix = ""
    else:
        gan_suffix = f"_{depth}_{f_base}_{sigma:4.2f}"
    if not np.isclose(image_weight, 1):
        gan_suffix += f"_iw{image_weight:.2f}"
    if not norm_params.get("train_on_gen", True):
        gan_suffix += "_ntog"
    else:
        if not np.isclose(image_gen_weight, 0.5):
            gan_suffix += f"_idg{image_gen_weight:4.2f}"
    if disc_n_conv != 3:
        gan_suffix += f"_nc{disc_n_conv}"
    if disc_filter_base != 32:
        gan_suffix += f"_fb{disc_filter_base}"
    if disc_type != "SimpleConv":
        gan_suffix += f"_{disc_type}"
    if not np.isclose(disc_start_lr, 0.05):
        gan_suffix += f"_{disc_start_lr:.5f}"
    return gan_suffix


def split_into_modalities(
    dataset, n_channels: int
) -> Dict[str, Dict[str, Union[List[str], str]]]:
    """Split the modalities of an existing dataset into its parts. It will replace
    the "image" part of the dictionary for each patient with multiple images.

    Parameters
    ----------
    dataset : Dict[str, Dict[str, Union[List[str], str]]]
        The dataset to process
    n_channels : int
        The number of channels

    Returns
    -------
    Dict[str, Dict[str, Union[List[str], str]]]
        The resulting dataset

    Raises
    ------
    ValueError
        _description_
    """
    experiment_dir = Path(os.environ["experiment_dir"])
    new_dict = {}
    for pat_name, data in tqdm(dataset.items(), desc="split quant"):
        new_dict[pat_name] = {k: v for k, v in data.items() if k not in ["image", "labels"]}
        new_dict[pat_name]["images"] = []
        new_dict[pat_name]["labels"] = data["labels"]
        image = None
        new_path = data["image"].parent / "channel_wise"
        new_path_abs = experiment_dir / new_path
        if not new_path_abs.exists():
            new_path_abs.mkdir()
        for i in range(n_channels):
            new_name = data["image"].name.replace(".nii.gz", f"_mod{i}.nii.gz")
            new_path_img_abs = experiment_dir / new_path / new_name
            if not new_path_img_abs.exists():
                if image is None:
                    image = sitk.ReadImage(str(experiment_dir / data["image"]))
                    if not image.GetNumberOfComponentsPerPixel() == n_channels:
                        raise ValueError("Image has the wrong number of channels.")
                image_channel = sitk.VectorIndexSelectionCast(image, i)
                sitk.WriteImage(image_channel, str(new_path_img_abs))
            new_dict[pat_name]["images"].append(new_path / new_name)

    return new_dict


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
            try:
                self.bot.send_message(text=message, chat_id=self.chat_id)
            except telegram.error.NetworkError:
                print("Sending of message failed, no internet.")

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
            try:
                self.bot.send_sticker(sticker=sticker, chat_id=self.chat_id)
            except telegram.error.NetworkError:
                print("Sending of message failed, no internet.")


def plot_metrics(data: pd.DataFrame, metrics: List[str]):
    """Plot the listed metrics including the validation

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot
    metrics : List[str]
        The metrics to plot
    """
    data = data.dropna(axis="columns")
    metrics_present = [t for t in metrics if f"val_{t}" in data]
    if len(metrics_present) == 0:
        return
    nrows = int(np.ceil(len(metrics_present) / 4))
    _, axes = plt.subplots(
        nrows=nrows,
        ncols=4,
        sharex=True,
        sharey=False,
        figsize=(16, nrows * 3.5),
    )
    for metric, ax in zip(metrics_present, axes.flat):
        plot_with_ma(ax, data, metric)
        ax.set_ylabel(metric)
    for ax in axes.flat[3 : 4 : len(metrics_present)]:
        ax.legend()
    for ax in axes.flat[len(metrics_present) :]:
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.close()


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
        for num, (field, y_label) in enumerate(
            zip(
                fields,
                names,
            )
        ):
            if field in res.columns:
                plot_with_ma(axes_line[num], res, field)
                axes_line[num].set_ylabel(y_label)
                last_present = max(last_present, num)
        axes_line[last_present].legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_disc_exp(res, experiments, prefixes, disc_type, channel=0):
    "Plot one channel over multiple experiments"
    res = res.query(f"channel=='{channel}'")
    if res.size == 0:
        return
    disc_list = [
        c[6 + len(disc_type) : -5]
        for c in res
        if c.startswith(f"disc_{disc_type}/") and c.endswith("loss") and len(c) > 17
    ]
    img_gen_list = [c for c in res.columns if c.startswith("disc_image_gen")]
    image_gen_pres = len(img_gen_list) > 0 and disc_type == "image"
    if image_gen_pres:
        ncols = 6
    else:
        ncols = 4
    _, axes_disc = plt.subplots(
        nrows=len(disc_list),
        ncols=ncols,
        sharex=True,
        sharey=False,
        figsize=(4 * ncols, len(disc_list) * 4),
    )

    for experiment, pre in zip(experiments, prefixes):
        res_exp = res.query(f"experiment == '{experiment}'")
        if res_exp.size == 0:
            continue
        if np.any(res_exp.epoch.value_counts() > 1):
            raise ValueError("Multiple data points for one epoch, check for duplicates")
        for disc, axes_line in zip(disc_list, axes_disc):
            disc_start = f"disc_{disc_type}/{disc}_"
            disc_metric = [c for c in res_exp.columns if c.startswith(disc_start)][
                0
            ].partition(disc_start)[-1]
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
            for num, (field, y_label) in enumerate(
                zip(
                    fields,
                    names,
                )
            ):
                if field in res_exp.columns:
                    plot_with_ma(
                        axes_line[num], res_exp, field, label_prefix=pre, dash_val=True
                    )
                    axes_line[num].set_ylabel(y_label)
                    last_present = max(last_present, num)
            axes_line[last_present].legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_with_ma(
    ax_ma: plt.Axes,
    data_frame: pd.DataFrame,
    field: str,
    window_size=5,
    label_prefix="",
    dash_val=False,
):
    """Plot a line with the value in the background and the moving average on top"""
    for val in ("", "val_"):
        for channel in range(3):
            df_channel = data_frame.query(f"channel == '{channel}'")
            if df_channel.size == 0:
                continue
            if dash_val and val == "val_":
                linestyle = "dashed"
            else:
                linestyle = "solid"
            plot = ax_ma.plot(
                df_channel.epoch,
                df_channel[val + field],
                alpha=0.4,
                linestyle=linestyle,
            )
            # plot with moving average
            color = plot[-1].get_color()
            if dash_val and val == "val_":
                color = color_not_val
                label = None
            else:
                color_not_val: str = color
                label = f"{label_prefix}{val}{channel}"
            ax_ma.plot(
                df_channel.epoch,
                np.convolve(
                    np.pad(
                        df_channel[val + field],
                        (window_size // 2 - 1 + window_size % 2, window_size // 2),
                        mode="edge",
                    ),
                    np.ones(window_size) / window_size,
                    mode="valid",
                ),
                label=label,
                linestyle=linestyle,
                color=color,
            )
