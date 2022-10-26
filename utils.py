"""
Miscellaneous functions
"""

import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import SimpleITK as sitk
import telegram
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
    image_gen_weight = norm_params.get("image_gen_weight", 1)
    disc_n_conv = norm_params.get("disc_n_conv", 3)
    disc_filter_base = norm_params.get("disc_filter_base", 32)
    disc_type = norm_params.get("disc_type", "SimpleConv")
    if depth == 3 and f_base == 16 and np.isclose(sigma, 1):
        gan_suffix = ""
    else:
        gan_suffix = f"_{depth}_{f_base}_{sigma:4.2f}"
    if norm_params.get("train_on_gen", False):
        gan_suffix += "_tog"
        if not np.isclose(image_gen_weight, 1):
            gan_suffix += f"_idg{image_gen_weight:4.2f}"
    if disc_n_conv != 3:
        gan_suffix += f"_nc{disc_n_conv}"
    if disc_filter_base != 32:
        gan_suffix += f"_fb{disc_filter_base}"
    if disc_type != "SimpleConv":
        gan_suffix += f"_{disc_type}"
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
