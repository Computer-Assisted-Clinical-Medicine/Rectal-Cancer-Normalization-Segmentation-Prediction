"""
Miscellaneous functions
"""

import itertools
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import telegram
import yaml
from tqdm import tqdm

from networks import ResNet
from SegClassRegBasis.architecture import DeepLabv3plus, DenseTiramisu, UNet
from SegClassRegBasis.evaluation import calculate_classification_metrics
from SegClassRegBasis.normalization import NORMALIZING
from SegClassRegBasis.utils import gather_results


def make_int(number):
    if number is None:
        return None
    else:
        return int(number)


def make_float(number):
    if number is None:
        return None
    else:
        return float(number)


def create_dataset(
    ignore=None,
    load_labels=True,
) -> Dict[str, Dict[str, Union[List[str], str]]]:
    """Create the dataset for segmentation or classification/regression

    Parameters
    ----------
    ignore : list, optional
        Which timepoints should be ignored, by default None
    load_labels : bool, optional
        If the labels should be used, then only timepoints with labels are loaded, by default True

    Returns
    -------
    Dict[str, Dict[str, Union[List[str], str]]]
        The dictionary containing images, labels and classification and regression data
    """
    if ignore is None:
        ignore = []
    patient_data = read_patient_data()
    data_dir = Path(os.environ["data_dir"])
    with open(data_dir / "segmented_images.yaml", encoding="utf8") as f:
        segmented_images = yaml.load(f, Loader=yaml.Loader)
    with open(data_dir / "images.yaml", encoding="utf8") as f:
        found_images = yaml.load(f, Loader=yaml.Loader)
    timepoints = pd.read_csv(data_dir / "timepoints.csv", sep=";", index_col=0)

    # create dict with all points. The names have the format:
    # patient_timepoint_l{label_number}_d{diffusion_number}
    dataset: Dict[str, Dict[str, Union[List[str], str]]] = {}
    for t_p, data in found_images.items():
        class_reg_labels = get_timepoint_patient_data(patient_data, timepoints, t_p)
        # either iterate the labeled image pairs or the T2 images
        if t_p in segmented_images:
            to_iterate_img = segmented_images[t_p]
        elif load_labels:
            continue
        else:
            if not "T2 axial" in data:
                continue
            to_iterate_img = [{"image": img} for img in data["T2 axial"]]
        for label_num, label_data in enumerate(to_iterate_img):
            found_data = found_images[t_p]
            t2_image = label_data["image"]
            image = t2_image.replace("Images", "Images registered and N4 corrected")
            if load_labels:
                label = label_data["labels"].replace(
                    "Images", "Images registered and N4 corrected"
                )
                label_dict = {
                    "labels": label,
                }
            else:
                label_dict = {}

            # look if all modalities are there
            if not "ADC axial original" in found_data:
                continue
            if not "Diff axial b800" in found_data:
                if "Diff axial b800 recalculated" in found_data:
                    b_name = "Diff axial b800 recalculated"
                else:
                    print("No b800 image found but an ADC image")
                    continue
            else:
                b_name = "Diff axial b800"

            for diff_num, (adc, b800) in enumerate(
                zip(found_data["ADC axial original"], found_data[b_name])
            ):
                name = f"{t_p}_l{label_num}_d{diff_num}"
                if name in ignore:
                    continue
                b800 = b800.replace("Images", "Images registered and N4 corrected")
                adc = adc.replace("Images", "Images registered and N4 corrected")
                img_dict = {
                    "images": [image, b800, adc],
                }
                dataset[name] = class_reg_labels | img_dict | label_dict

    return dataset


def get_timepoint_patient_data(patient_data, timepoints, t_p) -> dict:
    """Get the patient data from the dataframe and put it in the dictionary for
    use as labels

    Parameters
    ----------
    patient_data : pd.DataFrame
        The patient data
    timepoints : pd.DataFrame
        the timepoints
    t_p : str
        The name of the timepoint

    Returns
    -------
    dict
        The data as dict will classification and regression as keys
    """
    pat_data = patient_data.loc[int(t_p.partition("_")[0])].copy()
    # None is better for compatibility than pd.NA
    pat_data[pd.isna(pat_data)] = None

    tp_data = timepoints.loc[t_p]
    if tp_data.treatment_status == "before therapy":
        pre_therapy_t = pat_data.T_stage_pre_therapy
        pre_therapy_t_pat = pat_data.T_stage_patho
        pre_therapy_dworak = pat_data.dworak
    else:
        pre_therapy_t, pre_therapy_t_pat, pre_therapy_dworak = None, None, None
    if tp_data.treatment_status == "before OP":
        pre_op_t = pat_data.T_stage_pre_OP
        pre_op_t_pat = pat_data.T_stage_patho
        pre_op_dworak = pat_data.dworak
    else:
        pre_op_t, pre_op_t_pat, pre_op_dworak = None, None, None
    class_reg_labels = {
        "classification": {
            "sex": str(pat_data.sex),
            "dworak": make_int(pat_data.dworak),
            "T_stage_pre_therapy": make_int(pre_therapy_t),
            "T_stage_pre_therapy_patho": make_int(pre_therapy_t_pat),
            "T_stage_pre_OP": make_int(pre_op_t),
            "T_stage_pre_OP_patho": make_int(pre_op_t_pat),
            "dworak_pre_therapy": make_int(pre_therapy_dworak),
            "dworak_pre_OP": make_int(pre_op_dworak),
        },
        "regression": {
            "age": make_float(pat_data.age),
        },
    }

    return class_reg_labels


def generate_folder_name(parameters):
    """
    Make a name summarizing the hyperparameters.
    """
    epochs = parameters["train_parameters"]["epochs"]

    params = [
        parameters["architecture"].get_name() + str(parameters["dimensions"]) + "D",
    ]

    if parameters["architecture"] in (UNet, DenseTiramisu, DeepLabv3plus):
        if isinstance(parameters["loss"], str):
            loss_name = parameters["loss"]
        elif isinstance(parameters["loss"], dict):
            loss_name = parameters["loss"]["segmentation"]

        params.append(loss_name)

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
    elif parameters["architecture"] is ResNet:
        params.append(parameters["network_parameters"]["resnet_type"])
        if parameters["network_parameters"]["weights"] == "imagenet":
            params.append("imagenet")
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
    skip_edges = norm_params["skip_edges"]
    sigma = norm_params.get("smoothing_sigma", 1)
    image_weight = norm_params.get("image_weight", 1)
    image_gen_weight = norm_params.get("image_gen_weight", 1)
    disc_n_conv = norm_params.get("disc_n_conv", 3)
    disc_filter_base = norm_params.get("disc_filter_base", 32)
    disc_type = norm_params.get("disc_type", "SimpleConv")
    disc_start_lr = norm_params.get("disc_start_lr", 0.05)
    all_image = norm_params.get("all_image", False)
    init_norm_method = norm_params.get("init_norm_method", NORMALIZING.QUANTILE)
    train_on_segmentation = norm_params.get("train_on_segmentation", False)
    if depth == 3 and f_base == 16:
        gan_suffix = ""
    else:
        gan_suffix = f"_{depth}_{f_base}"
    if skip_edges:
        if not np.isclose(sigma, 1):
            gan_suffix += f"_{sigma:4.2f}"
    else:
        gan_suffix += "_n-skp"
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
    if all_image:
        gan_suffix += "_all_image"
    if init_norm_method is not NORMALIZING.QUANTILE:
        gan_suffix += f"_{init_norm_method.name}"
    if train_on_segmentation:
        gan_suffix += "_seg"
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
        if "labels" in data:
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


def read_patient_data() -> pd.DataFrame:
    """Read the patient data from the Dataset files

    Returns
    -------
    pd.DataFrame
        The patient data with the Patient ID as index
    """
    data_dir = Path(os.environ["data_dir"])
    # select images for processing
    patient_data_dir = data_dir / "Patient Data"
    patient_data_study = pd.read_csv(data_dir / "patients.csv", sep=";", index_col=0)
    patient_data_mannheim = pd.read_excel(
        patient_data_dir / "patient_data_mannheim.xlsx", index_col=1
    )

    patient_data = pd.DataFrame(
        index=list(patient_data_study.index) + list(patient_data_mannheim.index)
    )
    patient_data.index.name = "ID"

    _update_patient_data(
        patient_data,
        "dworak",
        patient_data_study,
        patient_data_mannheim,
        "post OP regressions Dworak",
        "Regression",
        pd.Int16Dtype(),
        {
            name: name[5]
            for name in patient_data_study["post OP regressions Dworak"].dropna().unique()
        },
    )

    _update_patient_data(
        patient_data,
        "sex",
        patient_data_study,
        patient_data_mannheim,
        "sex",
        "Geschl",
        pd.StringDtype(),
        {"männlich": "m", "weiblich": "f"},
        {"m": "m", "w": "f"},
    )

    t_stages_study = {f"T{t}": t for t in range(5)} | {"T4a": 4, "T4b": 4, "Tis": 1}
    # always use the higher stage
    t_stages_ma = {
        "Zwei Läsionen\noben T3\nunten T3": 3,
        "oben T2\nunten T3": 3,
        "1 bis 2": 2,
        "oben T2\nunten T1": 2,
        "oben T2\nunten T1 bis 2": 2,
        "fraglich T1,\nkaum noch abzugrenzen": 1,
        "T1 bis 2": 2,
    }
    _update_patient_data(
        patient_data,
        "T_stage_pre_therapy",
        patient_data_study,
        patient_data_mannheim,
        "T-stage_first_diagnosis",
        "praeT",
        pd.Int16Dtype(),
        t_stages_study,
        t_stages_ma,
    )

    _update_patient_data(
        patient_data,
        "T_stage_pre_OP",
        patient_data_study,
        patient_data_mannheim,
        "pre OP T-Stage",
        "postT",
        pd.Int16Dtype(),
        t_stages_study,
        t_stages_ma,
    )

    _update_patient_data(
        patient_data,
        "T_stage_patho",
        patient_data_study,
        patient_data_mannheim,
        "post OP pathological T-Stage",
        "pT",
        pd.Int16Dtype(),
        t_stages_study,
        t_stages_ma,
    )

    # add birthdays
    bday_study = pd.to_datetime(patient_data_study["birthday"])
    start_study = pd.to_datetime(patient_data_study["ct_start_date"])
    age_study = (start_study - bday_study) / np.timedelta64(1, "Y")

    bday_ma = pd.to_datetime(patient_data_mannheim["Geb"])
    start_ma = pd.to_datetime(patient_data_mannheim["MRT1"])
    age_ma = (start_ma - bday_ma) / np.timedelta64(1, "Y")

    patient_data["age"] = pd.concat([age_study, age_ma])

    return patient_data


def _update_patient_data(
    patient_data,
    name,
    df_study,
    df_ma,
    name_study,
    name_ma,
    dtype,
    func_study_dict=None,
    func_ma_dict=None,
):
    data_study = df_study[name_study].dropna()
    if func_study_dict is not None:

        data_study = data_study.replace(func_study_dict).astype(dtype)
    else:
        data_study = data_study.astype(dtype)
    data_ma = df_ma[name_ma].dropna()
    if func_ma_dict is not None:
        data_ma = data_ma.replace(func_ma_dict).astype(dtype)
    else:
        data_ma = data_ma.astype(dtype)
    data = pd.concat([data_study, data_ma]).astype(dtype)
    patient_data[name] = data
    patient_data[name] = patient_data[name].astype(dtype)


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


def gather_all_results(task="segmentation") -> pd.DataFrame:
    """Gather all results from the normalization experiment

    Parameters
    ----------
    tasks : str, optional
        Which tasks to analyze, by default "segmentation"

    Returns
    -------
    pd.DataFrame
        A dataframe with the results for each patient and timepoint for all tasks,
        locations and version
    """
    experiment_dir = Path(os.environ["experiment_dir"]) / "Normalization_Experiment"
    data_dir = Path(os.environ["data_dir"])
    timepoints = pd.read_csv(data_dir / "timepoints.csv", sep=";", index_col=0)
    with open(experiment_dir / "dataset.yaml", encoding="utf8") as f:
        orig_dataset = yaml.load(f, Loader=yaml.Loader)
    with open(experiment_dir / "dataset_class_reg.yaml", encoding="utf8") as f:
        orig_dataset_class_reg = yaml.load(f, Loader=yaml.Loader)
    collected_results = []
    for external in [True, False]:
        for postprocessed in [True, False]:
            for version in ["best", "final"]:
                if postprocessed and task != "segmentation":
                    continue
                loc_results = gather_results(
                    experiment_dir,
                    task=task,
                    external=external,
                    postprocessed=postprocessed,
                    version=version,
                    combined=False,
                )
                if loc_results is not None:
                    loc_results = loc_results.copy()
                    loc_results["external"] = external
                    loc_results["postprocessed"] = postprocessed
                    loc_results["version"] = version
                    collected_results.append(loc_results)

    results = pd.concat(collected_results)
    # defragment the frame
    results = results.copy()
    # reset index
    results.index = pd.RangeIndex(results.shape[0])
    # set timepoint
    results["timepoint"] = results["File Number"].apply(
        lambda x: "_".join(x.split("_")[:2])
    )
    results["network"] = results.name.apply(lambda x: x.split("-")[0])
    location_order = [
        "Frankfurt",
        "Regensburg",
        "Mannheim",
        "all",
        "Not-Frankfurt",
        "Not-Regensburg",
        "Not-Mannheim",
        "Frankfurt-all",
        "Regensburg-all",
        "Mannheim-all",
    ]
    results["train_location"] = pd.Categorical(
        results.exp_group_name.str.partition("_")[2], categories=location_order
    )
    assert results["train_location"].isna().sum() == 0
    # set treatment status
    mask = results.index[~results.timepoint.str.startswith("99")]
    results.loc[mask, "treatment_status"] = timepoints.loc[
        results.timepoint[mask]
    ].treatment_status.values
    mask = results["File Number"].str.contains("_1_l") & results.timepoint.str.startswith(
        "99"
    )
    results.loc[mask, "treatment_status"] = "before therapy"
    mask = results["File Number"].str.contains("_2_l") & results.timepoint.str.startswith(
        "99"
    )
    results.loc[mask, "treatment_status"] = "before OP"
    # set marker for before therapy
    results["before_therapy"] = results.treatment_status == "before therapy"

    for _, row in results.iterrows():
        assert row.normalization in row["name"], "normalization not in name"
    results.normalization = pd.Categorical(
        results.normalization, categories=sorted(results.normalization.unique())
    )
    results["from_study"] = ~results["File Number"].str.startswith("99")
    root = data_dir / "Images registered and N4 corrected"
    new_root = data_dir / "Images"
    # get image metadata
    param_list = []
    for number in results["File Number"].unique():
        if number in orig_dataset:
            images = [Path(img) for img in orig_dataset[number]["images"]]
        else:
            images = [Path(img) for img in orig_dataset_class_reg[number]["images"]]
        param_file = data_dir / images[0].parent / "acquisition_parameters.csv"
        param_file = new_root / param_file.relative_to(root)
        parameters = pd.read_csv(param_file, sep=";", index_col=0)
        assert isinstance(parameters, pd.DataFrame)
        t2_params = parameters.loc[parameters.filename == images[0].name].copy()
        t2_params["File Number"] = number
        t2_params["name"] = t2_params.index
        t2_params.set_index("File Number", inplace=True)
        param_list.append(t2_params)
    acquisition_params = pd.concat(param_list)
    # drop columns that are mostly empty or always the same
    for col in acquisition_params:
        num_na = acquisition_params[col].isna().sum()
        if num_na > acquisition_params.shape[0] // 2:
            acquisition_params.drop(columns=[col], inplace=True)
            continue
        same = (acquisition_params[col] == acquisition_params[col].iloc[0]).sum()
        if same > acquisition_params.shape[0] * 0.9:
            acquisition_params.drop(columns=[col], inplace=True)
            continue
    # correct pixel spacing
    def func(x):
        if "\\" in x:
            return float(x.split("\\")[0])
        else:
            return float(x[1:].split(",")[0])

    acquisition_params.pixel_spacing = acquisition_params["0028|0030"].apply(func)
    # correct location
    acquisition_params.loc[
        acquisition_params.index.str.startswith("99"), "location"
    ] = "Mannheim-not-from-study"

    # filter important parameters
    column_names = {
        "0008|1090": "model_name",
        "0018|0050": "slice_thickness",
        "0018|0080": "repetition_time",
        "0018|0095": "pixel_bandwidth",
        "0018|1314": "flip_angle",
        "0018|0081": "echo_time",
        "0018|0087": "field_strength",
        "pixel_spacing": "pixel_spacing",
        "location": "location",
    }
    acquisition_params = acquisition_params[column_names.keys()]
    acquisition_params.rename(columns=column_names, inplace=True)

    # set location
    results = results.merge(right=acquisition_params.location, on="File Number")
    return results, acquisition_params


def calculate_auc(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Calculate the AUC and other statistics for one task in the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with the columns {task}_probability_{lbl},
        {task}_ground_truth and {task}_top_prediction
    task : str
        The task name

    Returns
    -------
    pd.DataFrame
        The resulting metrics with the names as columns and the individual labels as rows
    """
    prob_prefix = f"{task}_probability_"
    prob_col_names = [c for c in df if prob_prefix in c]

    # remove missing values
    ground_truth = df[f"{task}_ground_truth"]
    probabilities = df[prob_col_names].values
    not_na = np.all(np.isfinite(probabilities), axis=-1)
    # if np.sum(~not_na):
    #     print(f"Nans found in {task}")
    mask = (ground_truth != -1) & not_na

    top_pred = df[f"{task}_top_prediction"][mask]
    probabilities = probabilities[mask]
    ground_truth = ground_truth[mask]
    if pd.api.types.is_numeric_dtype(ground_truth.dtype):
        ground_truth = ground_truth.astype(int)
    labels = np.array([val.partition(prob_prefix)[-1] for val in prob_col_names])

    if np.all(~mask):
        results = {}
    else:
        results = calculate_classification_metrics(
            prediction=top_pred,
            probabilities=probabilities,
            ground_truth=ground_truth,
            labels=labels,
        )

    res_df = pd.DataFrame(results)
    if pd.DataFrame(results).size > 0:
        res_df["task"] = task
        if not res_df.index.name == "label":
            res_df.set_index("label", inplace=True)
    return res_df


def hatched_histplot(
    data: pd.DataFrame,
    x: str,
    hue: str,
    multiple="stack",
    bins: Optional[np.ndarray] = None,
    hue_order: Optional[List] = None,
    legend=False,
    ax=Optional[plt.Axes],
    hatches: Optional[List] = None,
):
    """Add hatches to a plot, the api mirrors the one of the seaborn histogram."""
    groups = data.groupby(hue)[x]
    if bins is None:
        bins = np.linspace(start=groups.min().min(), stop=groups.max().max(), num=21)

    if multiple == "stack":
        hist_kwargs = dict(
            stacked=True, histtype="bar", alpha=0.8, edgecolor="gray", linewidth=0.5
        )
    else:
        raise ValueError(f"Multiple type {multiple} unknown.")

    if hue_order is None:
        hue_order = list(groups.groups)
    ax.hist(
        x=[groups.get_group(hue) for hue in hue_order],
        bins=bins,
        label=hue_order,
        hatch=["/"] * len(hue_order),
        **hist_kwargs,
    )
    # apply the hatching
    hatches = [
        "////",
        "\\\\\\\\",
        "xxxx",
        "ooo",
        "OO",
        "...",
        "||||",
        "----",
        "++++",
        "***",
    ]
    for container, hatch in zip(ax.containers, itertools.cycle(hatches)):
        for item in container.patches:
            item.set_hatch(hatch)
            item.set_edgecolor(plt.rcParams["hatch.color"])
    if legend:
        plt.legend()
