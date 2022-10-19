"""Normalize images using GANs"""
import os
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

import filelock
import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf
import yaml

import gan_networks
from SegClassRegBasis import config as cfg
from SegClassRegBasis.experiment import Experiment
from SegClassRegBasis.normalization import NORMALIZING, Normalization, all_subclasses
from SegClassRegBasis.preprocessing import preprocess_dataset
from SegClassRegBasis.segbasisnet import SegBasisNet


def make_normalization_dataset(
    modality: str, targets: Dict[str, str], column_names: Dict[str, str]
) -> Dict[str, Dict[str, Union[List[str], str, dict]]]:
    """Make a dataset for the training of the normalization. It will return
    all images with the given modality that are found in data_dir/images.yaml

    Parameters
    ----------
    modality : str
        The modality to use
    targets : Dict[str, str]
        The target columns to use, the key should be a name in the parameters.csv
        and the value should be reg or cat, depending on the task
    column_names : Dict[str, str]
        The new names for the columns, the keys should be the same as in target,
        only the values should be the new names

    Returns
    -------
    Dict[str, Dict[str, Union[List[str], str, dict]]]
        The dictionary containing the images and parameters
    """
    data_dir = Path(os.environ["data_dir"])
    # load data
    with open(data_dir / "images.yaml", encoding="utf8") as f:
        found_images = yaml.load(f, Loader=yaml.Loader)
    # create dict with all points. The names have the format:
    # patient_timepoint_l{label_number}_d{diffusion_number}
    dataset: Dict[str, Dict[str, Union[List[str], str, dict]]] = {}
    for timepoint, found_data in found_images.items():
        if not modality in found_data:
            continue
        for img_num, img in enumerate(found_data[modality]):
            name = f"{timepoint}_img{img_num}"
            image = img.replace("Images", "Images registered and N4 corrected")
            image_dir = (data_dir / img).parent
            param_file = image_dir / "acquisition_parameters.csv"
            params = pd.read_csv(param_file, sep=";", index_col=0)
            t2_name = Path(img).with_suffix("").with_suffix("").name
            missing_columns = [c for c in targets.keys() if c not in params]
            if len(missing_columns) > 0:
                raise ValueError(f"{missing_columns} not found")
            params_t2 = params.loc[t2_name, targets.keys()]
            cat_dict: Dict[str, str] = {}
            reg_dict: Dict[str, float] = {}
            for col, param_val in params_t2.iteritems():
                col_name = column_names.get(col, col)
                col_type = targets[col]
                if col_type == "reg":
                    reg_dict[col_name] = float(param_val)
                elif col_type == "cat":
                    cat_dict[col_name] = str(param_val).strip()
            # add location to scanner
            cat_dict["model_name"] = cat_dict["location"] + " - " + cat_dict["model_name"]
            dataset[name] = {
                "images": [image],
                "classification": cat_dict,
                "regression": reg_dict,
                "autoencoder": "image",
            }

    return dataset


def train_gan_normalization(
    timepoints_train: List[str],
    mod_num: int,
    preprocessed_dir: Path,
    experiment_group: Path,
    modality: str,
    n_epochs=200,
    depth=3,
    filter_base=16,
    min_max=False,
    smoothing_sigma=1.0,
    latent_weight=1,
    image_weight=1,
    image_gen_weight=1,
    skip_edges=True,
    latent=True,
    train_on_gen=False,
    gan_suffix="",
    **kwargs,
):
    """Train the GAN Normalization"""

    experiment_dir = Path(os.environ["experiment_dir"])
    data_dir = Path(os.environ["data_dir"])

    experiment_name = f"Train_Normalization_GAN_{mod_num}{gan_suffix}"
    exp_output_path = (
        experiment_group / f"Train_Normalization_GAN{gan_suffix}" / experiment_name
    )
    fold_dir = experiment_dir / exp_output_path / "fold-0"
    model_path = fold_dir / "models" / "model-final"
    model_path_rel = exp_output_path / "fold-0" / "models" / "model-final"

    start_lr = 0.05
    end_lr = 0.001
    lr_sd_type = "exponential_half"

    if model_path.exists():
        return model_path_rel

    targets = OrderedDict(
        {
            "0008|1090": "cat",  # : Manufacturer's Model Name
            "0018|0021": "cat",  # : Sequence Variant
            "0018|0050": "reg",  # : Slice Thickness
            "0018|0080": "reg",  # : Repetition Time
            "0018|0095": "reg",  # : Pixel Bandwidth
            "0018|0081": "reg",  # : Echo Time
            "0018|0087": "cat",  # : Magnetic Field Strength
            "pixel_spacing": "reg",
            "location": "cat",
        }
    )
    column_names = {
        "0008|1090": "model_name",
        "0018|0021": "sequence_variant",
        "0018|0050": "slice_thickness",
        "0018|0080": "repetition_time",
        "0018|0095": "pixel_bandwidth",
        "0018|0081": "echo_time",
        "0018|0087": "field_strength",
        "pixel_spacing": "pixel_spacing",
        "location": "location",
    }
    column_tasks = {column_names[field]: tsk for field, tsk in targets.items()}

    dataset = make_normalization_dataset(
        modality=modality, targets=targets, column_names=column_names
    )

    # set training files
    train_list = [key for key in dataset if key.partition("_img")[0] in timepoints_train]

    # define the parameters that are constant
    train_parameters = {
        "l_r": (lr_sd_type, start_lr, end_lr),
        "optimizer": "Adam",
        "epochs": n_epochs,
        "batch_size": 256,
        "in_plane_dimension": 128,
        # parameters for saving the best model
        "best_model_decay": 0.3,
        # scheduling parameters
        "early_stopping": False,
        # "patience_es": 30,
        "reduce_lr_on_plateau": False,
        # "patience_lr_plat": 20,
        # "factor_lr_plat": 0.5,
        "monitor": "val_generator/loss",
        "monitor_mode": "min",
        "save_best_only": False,
        # fine tuning parameters
        "finetune_epoch": None,
        "finetune_layers": None,
        "finetune_lr": None,
        # sampling parameters
        "samples_per_volume": 64,
        "background_label_percentage": 0.15,
        "percent_of_object_samples": 0,
        # Augmentation parameters
        "add_noise": False,
        "max_rotation": 0.1,
        "min_resolution_augment": 1.2,
        "max_resolution_augment": 0.9,
        # no tensorboard callback (slow)
        "write_tensorboard": False,
    }

    preprocessing_parameters = {
        "resample": True,
        "target_spacing": (1, 1, 3),
        "normalizing_method": NORMALIZING.QUANTILE,
        "normalization_parameters": {
            "lower_q": 0.05,
            "upper_q": 0.95,
        },
    }

    constant_parameters = {
        "train_parameters": train_parameters,
        "preprocessing_parameters": preprocessing_parameters,
        "dataloader_parameters": {"drop_remainder": True},
        "dimensions": 2,
    }

    # first discriminator just checks if the image looks good or not
    expanded_tasks = {}
    expanded_tasks["autoencoder"] = "autoencoder"
    discriminators = [
        {
            "name": "discriminator_real_fake",
            "input_type": "image",
            "discriminator_n_conv": 3,
            "loss": "MSE",
            "goal": "confuse",
        }
    ]
    # Other discriminators, use the median, there is nothing in the protocol
    reg_params_median = pd.DataFrame([d["regression"] for d in dataset.values()]).median()
    target_labels = reg_params_median.to_dict()
    target_labels.update(
        {
            "model_name": None,
            "sequence_variant": None,
            "field_strength": None,
            "location": None,
        }
    )
    input_types = {
        "model_name": "latent",
        "sequence_variant": "latent",
        "slice_thickness": "image",
        "repetition_time": "image",
        "pixel_bandwidth": "image",
        "flip_angle": "image",
        "echo_time": "image",
        "field_strength": "image",
        "pixel_spacing": "image",
        "location": "latent",
    }
    for col in column_names.values():
        if column_tasks[col] == "cat":
            disc_loss = "CEL"
            expanded_tasks[col] = "discriminator-classification"
        elif column_tasks[col] == "reg":
            disc_loss = "MSE"
            expanded_tasks[col] = "discriminator-regression"
        else:
            raise ValueError(f"Task {column_tasks[col]} unknown.")
        input_type = input_types[col]
        if input_type == "latent" and not latent:
            continue
        discriminators.append(
            {
                "name": col,
                "input_type": input_type,
                "loss": disc_loss,
                "goal": "predict",
                "target_labels": target_labels[col],
                "loss_weight": 0.01,
            }
        )

    hyperparameters: Dict[str, Any] = {
        **constant_parameters,
        "architecture": gan_networks.AutoencoderGAN,
        "network_parameters": {
            "depth": depth,
            "filter_base": filter_base,
            "skip_edges": skip_edges,
            "discriminators": discriminators,
            "regularize": (True, "L2", 0.001),
            "clip_value": 0.1,
            "variational": False,
            "train_on_gen": train_on_gen,
            "latent_weight": latent_weight,
            "image_weight": image_weight,
            "image_gen_weight": image_gen_weight,
            "smoothing_sigma": smoothing_sigma,
            "loss_parameters": {
                "NMI": {
                    "min_val": -1,
                    "max_val": 1,
                    "n_bins": 150,
                },
                "CON-OUT": {
                    "min_val": -1,
                    "max_val": 1,
                    "scaling": 10,
                },
            },
            # Discriminator arguments
            "disc_real_fake_optimizer": "Adam",
            "disc_real_fake_lr": (lr_sd_type, start_lr, end_lr),
            "disc_real_fake_n_conv": 3,
            "disc_image_optimizer": "Adam",
            "disc_image_lr": (lr_sd_type, start_lr, end_lr),
            "disc_image_n_conv": 3,
            "disc_latent_optimizer": "Adam",
            "disc_latent_lr": (lr_sd_type, start_lr, end_lr),
            "disc_latent_n_conv": 3,
        },
        "loss": {
            "autoencoder": "MSE",
            "classification": "CEL",
            "regression": "MSE",
            "discriminator-classification": "CEL",
            "discriminator-regression": "MSE",
        },
    }
    if min_max:
        hyperparameters["network_parameters"]["output_min"] = -1
        hyperparameters["network_parameters"]["output_max"] = 1

    # set a name for the preprocessing dir
    # so the same method will also use the same directory
    preprocessing_name = (
        hyperparameters["preprocessing_parameters"]["normalizing_method"].name
        + "_single_mod_"
        + modality.replace(" ", "_")
    )

    # set number of validation files
    hyperparameters["train_parameters"]["number_of_vald"] = max(len(train_list) // 20, 4)

    # preprocess data (only do that once for all experiments)
    exp_dataset = preprocess_dataset(
        data_set=dataset,
        num_channels=1,
        base_dir=experiment_dir,
        data_dir=data_dir,
        preprocessed_dir=preprocessed_dir / preprocessing_name,
        train_dataset=dataset.keys(),
        preprocessing_parameters=hyperparameters["preprocessing_parameters"],
    )

    # using nothing as training set
    cfg.data_train_split = 1

    fold_dir_rel = exp_output_path.parent / f"folds_norm_{modality.replace(' ', '_')}"

    exp = Experiment(
        hyper_parameters=hyperparameters,
        name=experiment_name,
        output_path_rel=exp_output_path,
        data_set=exp_dataset,
        crossvalidation_set=train_list,
        folds=1,
        seed=42,
        num_channels=1,
        folds_dir_rel=fold_dir_rel,
        tensorboard_images=True,
        versions=["final"],
        tasks=("autoencoder"),
        expanded_tasks=expanded_tasks,
    )

    lock_path = experiment_dir / exp_output_path / "lock_fold.txt.lock"
    if not lock_path.exists():
        with filelock.FileLock(lock_path, timeout=1):
            exp.train_fold(0)
    if lock_path.exists():
        lock_path.unlink()

    tf.keras.backend.clear_session()
    del exp

    return model_path_rel


class GAN_NORMALIZING(Enum):  # pylint:disable=invalid-name
    """The different normalization types
    To get the corresponding class, call get_class
    """

    GAN_DISCRIMINATORS = 0

    def get_class(self) -> Normalization:
        """Get the corresponding normalization class for an enum, it has to be a subclass
        of the Normalization class.

        Parameters
        ----------
        enum : NORMALIZING
            The enum

        Returns
        -------
        Normalization
            The normalization class

        Raises
        ------
        ValueError
            If the class was found for that enum
        """
        for norm_cls in all_subclasses(Normalization):
            if norm_cls.enum is self:
                return norm_cls
        raise ValueError(f"No normalization for {self.value}")


class GanDiscriminators(Normalization):
    """Use the trained networks to normalize the images"""

    enum = GAN_NORMALIZING.GAN_DISCRIMINATORS

    parameters_to_save = [
        "model_paths",
        "mod_num",
        "depth",
        "filter_base",
        "min_max",
        "smoothing_sigma",
        "latent_weight",
        "image_weight",
        "image_gen_weight",
        "skip_edges",
        "latent",
        "train_on_gen",
        "n_epochs",
    ]

    # make sure the parameters stay the same
    def __init__(
        self,
        model_paths: Path,
        mod_num: int,
        depth,
        filter_base,
        min_max,
        smoothing_sigma=1,
        latent_weight=1,
        image_weight=1,
        image_gen_weight=1,
        skip_edges=True,
        latent=True,
        train_on_gen=False,
        n_epochs=200,
        **kwargs,
    ) -> None:
        self.depth = depth
        self.filter_base = filter_base
        self.model_paths = model_paths
        self.mod_num = mod_num
        self.min_max = min_max
        self.smoothing_sigma = smoothing_sigma
        self.latent_weight = latent_weight
        self.image_weight = image_weight
        self.image_gen_weight = image_gen_weight
        self.skip_edges = skip_edges
        self.latent = latent
        self.train_on_gen = train_on_gen
        self.n_epochs = n_epochs
        self.model = None
        super().__init__(normalize_channelwise=False)

    def load_model(self):
        exp_dir = Path(os.environ["experiment_dir"])
        self.model = SegBasisNet(
            {"segmentation": "DICE"},
            model_path=exp_dir / self.model_paths[self.mod_num],
            is_training=False,
        )

    def normalize(self, image: sitk.Image) -> sitk.Image:
        """Apply the histogram matching to an image

        Parameters
        ----------
        image : sitk.Image
            The image

        Returns
        -------
        sitk.Image
            The normalized image
        """
        if self.model is None:
            self.load_model()

        image_np = sitk.GetArrayFromImage(image)

        pad_with = np.zeros((3, 2), dtype=int)
        div_h = 8
        min_p = 8
        for num in [1, 2]:
            size = image_np.shape[num]
            if size % 2 == 0:
                # and make sure have of the final size is divisible by divisible_by
                if div_h == 0:
                    pad_div = 0
                else:
                    pad_div = div_h - ((size // 2 + min_p) % div_h)
                pad_with[num] = min_p + pad_div
            else:
                # and make sure have of the final size is divisible by divisible_by
                if div_h == 0:
                    pad_div = 0
                else:
                    pad_div = div_h - (((size + 1) // 2 + min_p) % div_h)
                pad = min_p + pad_div
                # pad asymmetrical
                pad_with[num, 0] = pad + 1
                pad_with[num, 1] = pad

        image_np_padded = np.pad(image_np, pad_with)

        results = []
        for sample in image_np_padded:
            # add batch dimension
            sample_batch = sample.reshape((1,) + sample.shape)
            res = self.model.model(sample_batch)
            # make sure the result is a tuple
            if len(res) == 1:
                res = (res,)
            # convert to numpy
            res_np = res[0].numpy()
            results.append(res_np)

        # and concatenate them
        image_np_norm = np.concatenate(results, axis=0).squeeze()

        # remove the padding
        for num, (first, last) in enumerate(pad_with):
            image_np_norm = np.take(
                image_np_norm,
                indices=np.arange(first, image_np_norm.shape[num] - last),
                axis=num,
            )

        # and turn it back into an image
        image_normalized = sitk.GetImageFromArray(image_np_norm)
        image_normalized.CopyInformation(image)
        return image_normalized
