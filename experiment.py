"""
Class to run an experiment using user-defined hyperparameters.
"""
import copy
import logging
import os
from pathlib import Path, PurePath
from typing import Any, Dict, Iterable, List, Optional, OrderedDict

import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import yaml
from tqdm.autonotebook import tqdm

import evaluation
from seg_data_loader import ApplyLoader, SegLoader
from SegmentationNetworkBasis import config as cfg, segbasisnet
from SegmentationNetworkBasis import postprocessing

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# pylint: disable=too-many-lines


class Experiment:
    """Class to run an experiment using user-defined hyperparameters."""

    def __init__(
        self,
        name: str,
        hyper_parameters: dict,
        data_set: Dict[str, Dict[str, str]],
        crossvalidation_set: List,
        external_test_set: List = None,
        folds=5,
        versions=("best", "final"),
        seed=None,
        num_channels=1,
        output_path_rel=None,
        restart=False,
        reinitialize_folds=False,
        folds_dir_rel=None,
        tensorboard_images=False,
        tasks=("segmentation",),
        mapping=None,
    ):
        """Run experiments using a fixed set of hyperparameters

        Parameters
        ----------
            name : str
                Name of the experiment, is used for the folder name
            hyper_parameters : dict
                the hyperparameters that should be used (as soon as something is
                changed in between experiments, it is a hyperparameter)
            data_set : Dict[str, Dict[str, str]]
                Dict containing the dataset, for each entry, the key is used to
                reference that datapoint, the labels the labels file and the image
                key the images (all relative to the experiment dir)
            crossvalidation_set : List
                The list of images which should be used for training, validation and test
            external_test_set : Dict, optional
                The list of images if an external test set should also be used
            folds : int, optional
                The number of folds to use for validation, by default 5
            versions : Tuple, optional
                The versions of the network to use (final, best or both), by default ("best",)
            seed : int, optional
                the global seed, by default None
            num_channels: int, optional
                the number of channels in the data, default 1
            output_path_rel : str, optional
                path to write output in (relative to the experiment_dir env. variable),
                if None Experiments is used, by default None
            restart : bool, optional
                If already finished folds should be restarted, by default False
            reinitialize_folds : bool, optional
                If set to true, the split for the folds will be redone, by default False
            folds_dir_rel : str, optional
                Where the fold descripions should be saved (relative to the experiment_dir env. variable).
                All experiments sharing the same folds should have the same directory here, by default outputdir/folds
            tensorboard_images : bool, optional
                Wether to write images to tensorboard, takes a bit, so only for debugging, by default False
            tasks: tuple, optional
                Which tasks to perform, the choices are segmentation, classification, regression,
                by default ("segmentation",)
            mapping: dict, optional
                For classification and regression tasks, the mapping between the real values and
                training values, by default None
        """
        # do a deep copy of the parameters, because they contain lists and dicts
        self.hyper_parameters = copy.deepcopy(hyper_parameters)
        self.seed = seed
        self.name = name
        self.folds = folds
        self.num_channels = num_channels
        self.reinitialize_folds = reinitialize_folds
        self.data_set = data_set
        self.crossvalidation_set = np.array(crossvalidation_set)
        if external_test_set is not None:
            self.external_test_set: Optional[np.ndarray] = np.array(external_test_set)
            if self.external_test_set.size == 0:
                logger.warning("External test set is empty")
                self.external_test_set = None
        else:
            self.external_test_set = None

        if np.all([v in ("final", "best") for v in versions]):
            self.versions = versions
        else:
            raise ValueError(f"Version should be final or best, not {versions}.")

        # save the tasks
        self.tasks = tasks

        # get the environmental variables
        self.experiment_dir = Path(os.environ["experiment_dir"])

        # check input
        if len(self.crossvalidation_set) == 0:
            raise ValueError("Dataset is empty.")
        if cfg.number_of_vald * self.folds > self.crossvalidation_set.size:
            raise ValueError("Dataset to small for the specified folds.")

        for d_name in self.crossvalidation_set:
            if d_name not in self.data_set:
                raise KeyError(f"{d_name} not found in the data set")
            if "image" not in self.data_set[d_name]:
                raise ValueError(f"{d_name} does not have an image")
            img_path = self.experiment_dir / self.data_set[d_name]["image"]
            if not img_path.exists():
                raise FileNotFoundError(f"The image for {d_name} does not exist.")
            if "labels" not in self.data_set[d_name] and "segmentation" in self.tasks:
                raise ValueError(f"{d_name} does not have labels")
            if "segmentation" in self.tasks:
                lbl_path = self.experiment_dir / self.data_set[d_name]["labels"]
                if not lbl_path.exists():
                    raise FileNotFoundError(f"The labels file for {d_name} does not exist.")

        if self.external_test_set is not None:
            for d_name in self.external_test_set:
                if d_name not in self.data_set:
                    raise KeyError(f"{d_name} not found in the data set")
                if "image" not in self.data_set[d_name]:
                    raise ValueError(f"{d_name} does not have an image")
                img_path = self.experiment_dir / self.data_set[d_name]["image"]
                if not img_path.exists():
                    raise FileNotFoundError(f"The image for {d_name} does not exist.")

        # export datasets for classification and regression
        if mapping is None:
            self.mapping = None
            if "classification" in self.tasks:
                self.map_classification()
            if "regression" in self.tasks:
                self.map_regression()
        else:
            self.mapping = mapping

        self.train_dataset = self.convert_dataset(self.data_set)

        # create the expanded task list
        # TODO: support multiple Segmentation tasks
        self.expanded_tasks = OrderedDict()
        if "segmentation" in self.tasks:
            self.expanded_tasks["seg"] = "segmentation"
        if "classification" in self.tasks:
            for field in self.mapping["classification"].keys():
                self.expanded_tasks[field] = "classification"
        if "regression" in self.tasks:
            for field in self.mapping["regression"].keys():
                self.expanded_tasks[field] = "regression"
        self.hyper_parameters["network_parameters"]["tasks"] = self.expanded_tasks

        # get the label shapes, if the task is regression or segmentation
        if "classification" in self.tasks or "regression" in self.tasks:
            # for classification, output has the number of classes as shape
            class_shapes = tuple(
                max(val.values()) + 1 for val in self.mapping["classification"].values()
            )
            # for regression, output has shape 1
            reg_shapes = (1,) * len(self.mapping["regression"])
            self.hyper_parameters["network_parameters"]["label_shapes"] = (
                class_shapes + reg_shapes
            )

        if output_path_rel is None:
            self.output_path_rel = PurePath("Experiments", self.name)
        else:
            self.output_path_rel = PurePath(output_path_rel)
            if self.output_path_rel.is_absolute():
                raise ValueError("output_path_rel is an absolute path")

        # set the absolute path (which will not be exported)
        self.output_path = self.experiment_dir / Path(self.output_path_rel)

        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        logger.info("Set %s as output folder, all output will be there", self.output_path)

        # check for finetuning
        if not hasattr(self.hyper_parameters, "evaluate_on_finetuned"):
            self.hyper_parameters["evaluate_on_finetuned"] = False

        # set hyperparameterfile to store all hyperparameters
        self.experiment_file = self.output_path / "parameters.yaml"

        # set directory for folds
        if folds_dir_rel is None:
            self.folds_dir_rel = PurePath(self.output_path / "folds")
        else:
            self.folds_dir_rel = PurePath(folds_dir_rel)
            if self.folds_dir_rel.is_absolute():
                raise ValueError("folds_dir_rel is an absolute path")

        self.folds_dir = self.experiment_dir / Path(self.folds_dir_rel)

        if not self.folds_dir.exists():
            self.folds_dir.mkdir(parents=True)

        # set fold directory names
        self.fold_dir_names = [f"fold-{f}" for f in range(self.folds)]
        # set fold split file names
        self.datasets = []
        for fold in range(self.folds):
            # set paths
            train_csv = self.folds_dir / f"train-{fold}-{self.folds}.csv"
            vald_csv = self.folds_dir / f"vald-{fold}-{self.folds}.csv"
            test_csv = self.folds_dir / f"test-{fold}-{self.folds}.csv"
            self.datasets.append({"train": train_csv, "vald": vald_csv, "test": test_csv})
        # to the data split
        self.setup_folds(self.crossvalidation_set, overwrite=self.reinitialize_folds)

        self.restart = restart

        self.tensorboard_images = tensorboard_images

        # set postprocessing method
        self.postprocessing_method = postprocessing.keep_big_structures

        # export parameters
        self.export_experiment()

    def set_seed(self):
        """Set the seed in tensorflow and numpy"""
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def setup_folds(self, data_set: np.ndarray, overwrite=False):
        """Setup the split of the dataset. This will be done in the output_path
        and can be used by all experiments in that path.

        Parameters
        ----------
        data_set : np.ndarray
            The files in the dataset as np.ndarray
        overwrite : bool, optional
            IF this is true, existing files are overwritten, by default False
        """
        self.set_seed()

        if "number_of_vald" in self.hyper_parameters["train_parameters"]:
            cfg.number_of_vald = self.hyper_parameters["train_parameters"]["number_of_vald"]

        all_indices = np.random.permutation(range(0, data_set.size))
        # split the data into self.folds sections
        if self.folds > 1:
            test_folds = np.array_split(all_indices, self.folds)
        else:
            # otherwise, us cfg.data_train_split
            test_folds = all_indices[
                int(all_indices.size * cfg.data_train_split) :
            ].reshape(1, -1)

        for fold in range(0, self.folds):
            # test is the section
            test_indices = test_folds[fold]
            remaining_indices = np.setdiff1d(all_indices, test_folds[fold])
            # this orders the indices, so shuffle them again
            remaining_indices = np.random.permutation(remaining_indices)
            # number of validation is set in config
            vald_indices = remaining_indices[: cfg.number_of_vald]
            # the rest is used for training
            train_indices = remaining_indices[cfg.number_of_vald :]

            train_files = data_set[train_indices]
            vald_files = data_set[vald_indices]
            test_files = data_set[test_indices]

            # only write files if they do not exist or overwrite is true
            if not self.datasets[fold]["train"].exists() or overwrite:
                np.savetxt(
                    self.datasets[fold]["train"], train_files, fmt="%s", header="path"
                )
            if not self.datasets[fold]["vald"].exists() or overwrite:
                np.savetxt(self.datasets[fold]["vald"], vald_files, fmt="%s", header="path")
            if not self.datasets[fold]["test"].exists() or overwrite:
                np.savetxt(self.datasets[fold]["test"], test_files, fmt="%s", header="path")

    def _set_parameters(self):
        """This function will set up the shapes in the cfg module so that they
        will run on the current GPU and will set the parameters for the
        augmentation.
        """

        # use this to write less
        hp_train: dict = self.hyper_parameters["train_parameters"]

        if "number_of_vald" in hp_train:
            cfg.number_of_vald = hp_train["number_of_vald"]

        # set sampling parameters
        cfg.percent_of_object_samples = hp_train["percent_of_object_samples"]
        cfg.samples_per_volume = hp_train["samples_per_volume"]
        cfg.background_label_percentage = hp_train["background_label_percentage"]

        assert "batch_size" in hp_train

        # set noise parameters
        # noise
        cfg.add_noise = hp_train["add_noise"]
        if cfg.add_noise:
            cfg.noise_typ = hp_train["noise_typ"]
            cfg.standard_deviation = hp_train["standard_deviation"]
            cfg.mean_poisson = hp_train["mean_poisson"]
        # rotation
        cfg.max_rotation = hp_train["max_rotation"]
        # scale change
        cfg.min_resolution_augment = hp_train["min_resolution_augment"]
        cfg.max_resolution_augment = hp_train["max_resolution_augment"]

        cfg.num_channels = self.num_channels
        assert "in_plane_dimension" in hp_train
        p_dim = int(hp_train["in_plane_dimension"])
        cfg.train_dim = p_dim
        z_dim = hp_train.get("number_slices", 32)  # the resolution in z-direction
        cfg.num_slices_train = z_dim

        cfg.batch_size_train = hp_train["batch_size"]
        if cfg.batch_size_train < 4:
            raise Warning("Batch size is below 4, which is pretty small.")
        cfg.batch_size_valid = cfg.batch_size_train

        # set shape according to the dimension
        dim = self.hyper_parameters["dimensions"]
        n_cls_seg = cfg.num_classes_seg
        if dim == 2:
            # set shape
            cfg.train_input_shape = [p_dim, p_dim, self.num_channels]
            cfg.train_label_shape = [p_dim, p_dim, n_cls_seg]

            # there are 10-30 layers per image containing foreground data. Half the
            # samples are taken from the foreground, so take about 64 samples
            # to cover all the foreground pixels at least once on average, but
            cfg.samples_per_volume = 64
            logger.debug(
                "   Train Shapes: %s (input), %s (labels)",
                cfg.train_input_shape,
                cfg.train_label_shape,
            )
        elif dim == 3:
            cfg.train_input_shape = [
                z_dim,
                p_dim,
                p_dim,
                self.num_channels,
            ]
            cfg.train_label_shape = [
                z_dim,
                p_dim,
                p_dim,
                n_cls_seg,
            ]

            cfg.samples_per_volume = 8
            logger.debug(
                "   Train Shapes: %s (input), %s (labels)",
                cfg.train_input_shape,
                cfg.train_label_shape,
            )

        # set the valid batch size
        cfg.batch_size_valid = cfg.batch_size_train
        # see if the batch size is bigger than the validation set
        if cfg.samples_per_volume * cfg.number_of_vald <= cfg.batch_size_valid:
            cfg.batch_size_valid = cfg.samples_per_volume * cfg.number_of_vald

        cfg.batch_capacity_train = 4 * cfg.samples_per_volume

    def map_classification(self):
        """Map the categorical columns to numerical values, which will be used
        for one-hot encoding
        """
        classification_df = pd.DataFrame.from_dict(
            {k: v["classification"] for k, v in self.data_set.items()}, orient="index"
        )
        if self.mapping is None:
            self.mapping = {}
        self.mapping["classification"] = OrderedDict()
        for col in classification_df:
            col_data = pd.Categorical(classification_df[col])
            self.mapping["classification"][col] = {
                cat: i for i, cat in enumerate(col_data.categories)
            }

    def map_regression(self):
        """Map the regression columns to numerical values between 0 and 1, which
        will be used as network output
        """
        classification_df = pd.DataFrame.from_dict(
            {k: v["regression"] for k, v in self.data_set.items()}, orient="index"
        )
        if self.mapping is None:
            self.mapping = {}
        self.mapping["regression"] = OrderedDict()
        for col in classification_df:
            col_data = classification_df[col]
            col_min = col_data.min()
            col_max = col_data.max()
            self.mapping["regression"][col] = {
                0: float(col_min),
                1: float(col_max),
            }

    def convert_dataset(self, dataset: Dict) -> Dict:
        """Convert the dataset to a format that can be used to train the neural
        network. The classification columns will be converted to one-hot encoding
        and the regression columns to a normalized numeric value. The image and
        labels will be kept as is.

        Parameters
        ----------
        dataset : Dict
            The dataset to convert. There can be entries for image, labels, classification
            abd regression. The last Two should be a dict themselves with the individual
            categories as keys.

        Returns
        -------
        Dict
            The converted dict with the same keys as the input. Only classification
            and regression will be changed to a list of numpy arrays.
        """
        # generate the classification mapping
        class_map = OrderedDict()
        for feature, values in self.mapping["classification"].items():
            matrix = np.eye(max(values.values()) + 1)
            class_map[feature] = {k: matrix[v] for k, v in values.items()}
        # and the regression mapping
        reg_map = OrderedDict()
        for feature, values in self.mapping["regression"].items():
            reg_map[feature] = scipy.interpolate.interp1d(
                list(values.values()), list(values.keys())
            )

        train_dataset = {}
        for patient, data in dataset.items():
            train_dataset[patient] = {}
            if "image" in data:
                train_dataset[patient]["image"] = data["image"]
            if "labels" in data:
                train_dataset[patient]["labels"] = data["labels"]
            if "classification" in data:
                train_dataset[patient]["classification"] = [
                    f_map[data["classification"][f_name]]
                    for f_name, f_map in class_map.items()
                ]
            if "regression" in data:
                train_dataset[patient]["regression"] = [
                    f_map(data["regression"][f_name]) for f_name, f_map in reg_map.items()
                ]
        return train_dataset

    def training(self, folder_name: str, train_files: List, vald_files: List):
        """Do the actual training

        Parameters
        ----------
        folder_name : str
            Training output will be in the output path in this subfolder
        train_files : List
            List of training files as string
        vald_files : List
            List of validation files as string
        """
        tf.keras.backend.clear_session()

        # set data dir
        cfg.data_base_dir = self.experiment_dir.resolve()

        # generate loader
        training_dataset = SegLoader(
            name="training_loader",
            file_dict=self.train_dataset,
            frac_obj=self.hyper_parameters["train_parameters"]["percent_of_object_samples"],
        )(
            train_files,
            batch_size=cfg.batch_size_train,
            n_epochs=self.hyper_parameters["train_parameters"]["epochs"],
        )
        validation_dataset = SegLoader(
            mode=SegLoader.MODES.VALIDATE,
            name="validation_loader",
            file_dict=self.train_dataset,
            frac_obj=self.hyper_parameters["train_parameters"]["percent_of_object_samples"],
        )(
            vald_files,
            batch_size=cfg.batch_size_valid,
            n_epochs=self.hyper_parameters["train_parameters"]["epochs"],
        )

        # just use one sample with the foreground class using the validation files
        if "segmentation" in self.tasks:
            frac_obj_val = 1
        else:
            frac_obj_val = 0
        if self.tensorboard_images:
            visualization_dataset = SegLoader(
                name="visualization",
                file_dict=self.train_dataset,
                frac_obj=frac_obj_val,
                samples_per_volume=5,
                shuffle=False,
            )(
                vald_files,
                batch_size=cfg.batch_size_train,
                n_epochs=self.hyper_parameters["train_parameters"]["epochs"],
            )
        else:
            visualization_dataset = None

        # only do a graph for the first fold
        write_graph = folder_name == "fold-0"

        net: segbasisnet.SegBasisNet = self.hyper_parameters["architecture"](
            loss_name=self.hyper_parameters["loss"],
            # add initialization parameters
            **self.hyper_parameters["network_parameters"],
        )
        # Train the network with the dataset iterators
        logger.info("Started training of %s", folder_name)
        net.train(
            logs_path=str(self.output_path),
            folder_name=folder_name,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            visualization_dataset=visualization_dataset,
            write_graph=write_graph,
            visualize_labels="segmentation" in self.tasks,
            # add training parameters
            **(self.hyper_parameters["train_parameters"]),
        )

    def applying(
        self, folder_name: str, test_files: Iterable, apply_name="apply", version="best"
    ):
        """Apply the trained network to the test files

        Parameters
        ----------
        folder_name : str
            Training output will be in the output path in this subfolder
        test_files : Iterable
            Iterable of test files as string
        apply_name : str, optional
            The subfolder where the evaluated files are stored, by default apply
        version : str, optional
            The version to use (final or best), by default "best"
        """
        tf.keras.backend.clear_session()

        # set data dir
        cfg.data_base_dir = self.experiment_dir

        testloader = ApplyLoader(name="test_loader", file_dict=self.train_dataset)

        net = self.hyper_parameters["architecture"](
            self.hyper_parameters["loss"],
            is_training=False,
            model_path=str(self.output_path / folder_name / "models" / f"model-{version}"),
            **(self.hyper_parameters["network_parameters"]),
        )

        logger.info("Started applying %s to test datset.", folder_name)

        apply_path = self.output_path / folder_name / apply_name
        if not apply_path.exists():
            apply_path.mkdir()
        for file in tqdm(
            test_files, desc=f'{folder_name} ({apply_name.replace("_", " ")})', unit="file"
        ):
            f_name = Path(file).name

            # do inference
            result_npz = apply_path / f"prediction-{f_name}-{version}.npz"
            result_image = apply_path / f"prediction-{f_name}-{version}{cfg.file_suffix}"
            if not result_npz.exists():
                net.apply(
                    version=version,
                    application_dataset=testloader,
                    filename=file,
                    apply_path=apply_path,
                )

            # postprocess the image
            if "segmentation" in self.tasks:
                postprocessed_image = (
                    apply_path
                    / f"prediction-{f_name}-{version}-postprocessed{cfg.file_suffix}"
                )
                if not postprocessed_image.exists():
                    self.postprocess(result_image, postprocessed_image)

        tf.keras.backend.clear_session()

    def postprocess(self, unprocessed, processed):
        """Postprocess the label images with the method set as postprocessing
        method.

        Parameters
        ----------
        unprocessed : str
            The path of the unprocessed image
        processed : str
            The path of the processed image
        """
        self.postprocessing_method(unprocessed, processed)

    def evaluate_fold(
        self, folder_name, test_files, name="test", apply_name="apply", version="best"
    ):
        """Evaluate the files generated by the network

        Parameters
        ----------
        folder_name : str
            Training output will be in the output path in this subfolder
        test_files : List
            List of test files as string
        name : str, optional
            The name of the test set, by default test
        apply_name : str, optional
            The subfolder where the evaluated files are stored, by default apply
        version : str, optional
            The version of the results to use, by default best
        """

        logger.info("Start evaluation of %s.", folder_name)

        apply_path = self.output_path / folder_name / apply_name
        if not apply_path.exists():
            raise FileNotFoundError(f"The apply path {apply_path} does not exist.")

        for task in self.tasks:

            # only evaluate postprocessed files for segmentation
            if task in ("classification", "regression"):
                if "postprocessed" in version:
                    continue

            eval_name = f"evaluation-{folder_name}-{version}_{name}-{task}.csv"
            eval_file_path = self.output_path / folder_name / eval_name

            # remember the results
            results = []

            for file in test_files:
                prediction_path = apply_path / f"prediction-{file}-{version}.npz"
                if not prediction_path.exists():
                    raise FileNotFoundError(f"{prediction_path} was not found.")
                if task == "segmentation":
                    file_metrics = self.evaluate_segmentation(file, prediction_path)
                elif task == "classification":
                    file_metrics = self.evaluate_classification(file, prediction_path)
                elif task == "regression":
                    file_metrics = self.evaluate_regression(file, prediction_path)
                else:
                    raise ValueError(f"Task {task} unknown")
                if file_metrics is not None:
                    file_metrics["File Number"] = file
                    results.append(file_metrics)

            # write evaluation results
            if len(results) == 0:
                continue
            results = pd.DataFrame(results)
            results.set_index("File Number", inplace=True)
            results.to_csv(eval_file_path, sep=";")

    def evaluate_segmentation(self, file: str, prediction_path: Path) -> Dict[str, Any]:
        """Evaluate the segmentation of a single image

        Parameters
        ----------
        file : str
            The file identifier to analyse
        prediction_path : Path
            The path of the prediction

        Returns
        -------
        Dict[str, Any]
            The resulting metrics as a dictionary with each metric as one entry
        """
        # see that the labels are there
        if not "labels" in self.data_set[file]:
            logger.info("No labels found for %s", file)
            return None
        label_path = self.experiment_dir / self.data_set[file]["labels"]
        if not label_path.exists():
            logger.info("Label %s does not exists. It will be skipped", label_path)
            raise FileNotFoundError(f"Labels {label_path} not found.")
        # do the evaluation
        try:
            result_metrics = evaluation.evaluate_segmentation_prediction(
                str(prediction_path), str(label_path)
            )
            logger.info("        Finished Evaluation for %s", file)
        except RuntimeError as err:
            logger.exception("Evaluation failed for %s, %s", file, err)
        return result_metrics

    def evaluate_classification(self, file: str, prediction_path: Path) -> Dict[str, Any]:
        """Evaluate the classification of a single image

        Parameters
        ----------
        file : str
            The file identifier to analyse
        prediction_path : Path
            The path of the prediction

        Returns
        -------
        Dict[str, Any]
            The resulting metrics as a dictionary with each metric as one entry
        """
        # get the dictionaries
        class_dict = self.data_set[file]["classification"]
        results = np.load(prediction_path)
        mapping = self.mapping["classification"]
        result_metrics = {}
        # do the evaluation
        for col_name, map_dict in mapping.items():
            results_col = results[col_name]
            ground_truth = map_dict[class_dict[col_name]]
            col_metrics = evaluation.evaluate_classification(results_col, ground_truth)
            for key, value in col_metrics.items():
                result_metrics[f"{col_name}_{key}"] = value
        return result_metrics

    def evaluate_regression(self, file: str, prediction_path: Path) -> Dict[str, Any]:
        """Evaluate the regression of a single image

        Parameters
        ----------
        file : str
            The file identifier to analyse
        prediction_path : Path
            The path of the prediction

        Returns
        -------
        Dict[str, Any]
            The resulting metrics as a dictionary with each metric as one entry
        """
        # get the dictionaries
        class_dict = self.data_set[file]["regression"]
        results = np.load(prediction_path)
        mapping = self.mapping["regression"]
        result_metrics = {}
        # do the evaluation
        for col_name, map_dict in mapping.items():
            results_col = results[col_name]
            ground_truth = float(
                scipy.interpolate.interp1d(list(map_dict.values()), list(map_dict.keys()))(
                    class_dict[col_name]
                )
            )
            col_metrics = evaluation.evaluate_regression(results_col, ground_truth)
            for key, value in col_metrics.items():
                result_metrics[f"{col_name}_{key}"] = value
        return result_metrics

    def run_all_folds(self):
        """This is just a wrapper for run_fold and runs it for all folds"""
        self.set_seed()

        for (fold,) in range(0, self.folds):
            self.run_fold(fold)

    def run_fold(self, fold: int):
        """Run the training and evaluation for all folds

        Parameters
        ----------
        fold : int
            The number of the fold
        """

        folder_name = self.fold_dir_names[fold]
        folddir = self.output_path / folder_name
        logger.info("workingdir is %s", folddir)

        tqdm.write(
            f"Starting with {self.name} {folder_name} (Fold {fold+1} of {self.folds})"
        )

        train_files = np.loadtxt(self.datasets[fold]["train"], dtype="str", delimiter=",")
        vald_files = np.loadtxt(self.datasets[fold]["vald"], dtype="str", delimiter=",")
        test_files = np.loadtxt(self.datasets[fold]["test"], dtype="str", delimiter=",")

        if not folddir.exists():
            folddir.mkdir()

        cfg.num_files = len(train_files)
        assert cfg.number_of_vald == len(vald_files), "Wrong number of valid files"

        logger.info(
            "  Data Set %s: %s  train cases, %s  test cases, %s vald cases",
            fold,
            train_files.size,
            vald_files.size,
            test_files.size,
        )

        self._set_parameters()

        epoch_samples = cfg.samples_per_volume * cfg.num_files
        if not epoch_samples % cfg.batch_size_train == 0:
            print(
                "Sample Number not divisible by batch size, epochs will run a little bit short."
            )
        if cfg.batch_size_train > cfg.samples_per_volume * cfg.num_files:
            print("Reduce batch size to epoch size")
            cfg.batch_size_train = cfg.samples_per_volume * cfg.num_files

        # try the actual training
        model_result = folddir / "models" / "model-final"
        if self.restart is False and model_result.exists():
            tqdm.write("Already trained, skip training.")
            logger.info("Already trained, skip training.")
        else:
            self.training(folder_name, train_files, vald_files)

        for version in self.versions:
            # do the application and evaluation
            eval_file_path = folddir / f"evaluation-{folder_name}-{version}_test.csv"
            if eval_file_path.exists():
                tqdm.write(f"Already evaluated {version}, skip evaluation.")
                logger.info("Already evaluated %s, skip evaluation.", version)
            else:
                self.applying(folder_name, test_files, version=version)
                self.evaluate_fold(folder_name, test_files, version=version)

            # evaluate the postprocessed files
            eval_file_path = (
                folddir / f"evaluation-{folder_name}-{version}-postprocessed_test.csv"
            )
            if not eval_file_path.exists():
                self.evaluate_fold(
                    folder_name, test_files, version=f"{version}-postprocessed"
                )

            # evaluate the external set if present
            if self.external_test_set is not None:
                ext_eval = (
                    folddir / f"evaluation-{folder_name}-{version}_external_testset.csv"
                )
                if ext_eval.exists():
                    tqdm.write("Already evaluated on external set, skip evaluation.")
                    logger.info("Already evaluated on external set, skip evaluation.")
                else:
                    self.applying(
                        folder_name,
                        self.external_test_set,
                        apply_name="apply_external_testset",
                        version=version,
                    )
                    self.evaluate_fold(
                        folder_name,
                        self.external_test_set,
                        name="external_testset",
                        apply_name="apply_external_testset",
                        version=version,
                    )
                # also evaluate the postprocessed version
                ext_eval = (
                    folddir
                    / f"evaluation-{folder_name}-{version}-postprocessed_external_testset.csv"
                )
                if not ext_eval.exists():
                    self.evaluate_fold(
                        folder_name,
                        self.external_test_set,
                        name="external_testset",
                        apply_name="apply_external_testset",
                        version=f"{version}-postprocessed",
                    )

        tqdm.write(
            f"Finished with {self.name} {folder_name} (Fold {fold+1} of {self.folds})"
        )

    def evaluate(self, name="test"):
        """Evaluate the training over all folds

        name : str, optional
            The name of the set to evaluate, by default test

        Raises
        ------
        FileNotFoundError
            If and eval file was not found, most likely because the training failed
            or is not finished yet
        """

        # use all versions plus their postprocessed versions
        for version in [v + p for p in ["", "-postprocessed"] for v in self.versions]:
            for task in self.tasks:
                # set eval files
                eval_files = []
                for f_name in self.fold_dir_names:
                    eval_files.append(
                        self.output_path
                        / f_name
                        / f"evaluation-{f_name}-{version}_{name}-{task}.csv"
                    )
                if not np.all([f.exists() for f in eval_files]):
                    raise FileNotFoundError("Eval file not found")
                # combine previous evaluations
                output_path = self.output_path / f"results_{name}_{version}_{task}"
                if not output_path.exists():
                    output_path.mkdir()
                evaluation.combine_evaluation_results_from_folds(output_path, eval_files)
                # make plots
                if task == "segmentation":
                    evaluation.make_boxplot_graphic(
                        output_path, output_path / "evaluation-all-files.csv"
                    )

    def evaluate_external_testset(self):
        """evaluate the external testset, this just call evaluate for this set."""
        self.evaluate(name="external_testset")

    def export_experiment(self, overwrite=False):
        """Export the experiment ot disk. This can be used to start it again
        on a distributed system.

        Parameters
        ----------
        overwrite : bool, optional
            If the existing file should be overwritten, by default False
        """
        if not overwrite and self.experiment_file.exists():
            return
        experiment_dict = {
            "name": self.name,
            "hyper_parameters": self.hyper_parameters,
            "folds": self.folds,
            "seed": self.seed,
            "num_channels": self.num_channels,
            "output_path_rel": self.output_path_rel,
            "restart": self.restart,
            "reinitialize_folds": self.reinitialize_folds,
            "folds_dir_rel": self.folds_dir_rel,
            "tensorboard_images": self.tensorboard_images,
            "crossvalidation_set": [str(f) for f in self.crossvalidation_set],
            "data_set": self.data_set,
            "versions": self.versions,
            "tasks": self.tasks,
            "mapping": self.mapping,
        }
        if hasattr(self, "external_test_set"):
            if self.external_test_set is not None:
                ext_set = [str(f) for f in self.external_test_set]
                experiment_dict["external_test_set"] = ext_set
        with open(self.experiment_file, "w", encoding="utf8") as f:
            yaml.dump(experiment_dict, f, sort_keys=False)
        return

    @classmethod
    def from_file(cls, file):
        """Open an existing experiment file.

        Parameters
        ----------
        file : PathLike
            The file to open

        Returns
        -------
        Experiment
            The experiment as object
        """
        with open(file, "r", encoding="utf8") as f:
            parameters = yaml.load(f, Loader=yaml.Loader)
        return cls(**parameters)
