"""
Miscellaneous functions
"""

import os
from pathlib import Path
from typing import Dict, List, Union

import SimpleITK as sitk

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
        norm_params = parameters["preprocessing_parameters"]["normalization_parameters"]
        depth = norm_params["depth"]
        f_base = norm_params["filter_base"]
        sigma = norm_params["smoothing_sigma"]
        if depth == 3 and f_base == 16 and sigma == 1:
            gan_postfix = ""
        else:
            gan_postfix = f"_{depth}_{f_base}_{sigma:4.2f}"
        norm_name += gan_postfix
    params.append(str(norm_name))

    # object fraction
    params.append(
        f'obj_{int(parameters["train_parameters"]["percent_of_object_samples"]*100):03d}%'
    )

    # add epoch number
    params.append(str(epochs))

    folder_name = "-".join(params)

    return folder_name


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
    for pat_name, data in dataset.items():
        new_dict[pat_name] = {k: v for k, v in data.items() if k not in ["image", "labels"]}
        new_dict[pat_name]["images"] = []
        new_dict[pat_name]["labels"] = data["labels"]
        image = sitk.ReadImage(str(experiment_dir / data["image"]))
        if not image.GetNumberOfComponentsPerPixel() == n_channels:
            raise ValueError("Image has the wrong number of channels.")
        new_path = data["image"].parent / "channel_wise"
        new_path_abs = experiment_dir / new_path
        if not new_path_abs.exists():
            new_path_abs.mkdir()
        for i in range(n_channels):
            new_name = data["image"].name.replace(".nii.gz", f"_mod{i}.nii.gz")
            new_path_img_abs = experiment_dir / new_path / new_name
            if not new_path_img_abs.exists():
                image_channel = sitk.VectorIndexSelectionCast(image, i)
                sitk.WriteImage(image_channel, str(new_path_img_abs))
            new_dict[pat_name]["images"].append(new_path / new_name)

    return new_dict
