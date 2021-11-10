"""
Class to run an experiment using user-defined hyperparameters.
"""
import copy
import logging
import os
import sys
from pathlib import Path, PurePath
from typing import Dict, Iterable, List, Optional

import GPUtil
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tqdm.autonotebook import tqdm

import evaluation
from seg_data_loader import ApplyLoader, SegLoader
from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis import postprocessing
from utils import export_slurm_job

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
            self.tensorboard_images : bool, optional
                Wether to write images to tensorboard, takes a bit, so only for debugging, by default False
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
            if "labels" not in self.data_set[d_name]:
                raise ValueError(f"{d_name} does not have labels")
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
        hp_train = self.hyper_parameters["train_parameters"]

        # set sampling parameters
        cfg.percent_of_object_samples = hp_train["percent_of_object_samples"]
        cfg.samples_per_volume = hp_train["samples_per_volume"]
        cfg.background_label_percentage = hp_train["background_label_percentage"]

        # determine batch size if not set
        if "batch_size" not in hp_train:
            batch_size = self.estimate_batch_size()
            hp_train["batch_size"] = int(batch_size)

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
        if self.hyper_parameters["architecture"].get_name() == "UNet":
            cfg.train_dim = 128  # the resolution in plane
        elif self.hyper_parameters["architecture"].get_name() == "DenseTiramisu":
            cfg.train_dim = 64  # the resolution in plane
        elif self.hyper_parameters["architecture"].get_name() == "DeepLabv3plus":
            cfg.train_dim = 256  # the resolution in plane
        else:
            raise ValueError("Unrecognized network")
        cfg.num_slices_train = 32  # the resolution in z-direction

        cfg.batch_size_train = hp_train["batch_size"]
        cfg.batch_size_valid = cfg.batch_size_train

        # set shape according to the dimension
        dim = self.hyper_parameters["dimensions"]
        if dim == 2:
            # set shape
            cfg.train_input_shape = [cfg.train_dim, cfg.train_dim, cfg.num_channels]
            cfg.train_label_shape = [cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]

            # set sample numbers
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
            # set shape
            # if batch size too small, decrease z-extent
            if (
                cfg.batch_size_train < 4
                and self.hyper_parameters["architecture"].get_name() == "UNet"
            ):
                cfg.num_slices_train = cfg.num_slices_train // 2
                cfg.batch_size_train = cfg.batch_size_train * 2
                # if still to small, decrease patch extent in plane
                if cfg.batch_size_train < 4:
                    cfg.train_dim = cfg.train_dim // 2
                    cfg.batch_size_train = cfg.batch_size_train * 2
            cfg.train_input_shape = [
                cfg.num_slices_train,
                cfg.train_dim,
                cfg.train_dim,
                cfg.num_channels,
            ]
            cfg.train_label_shape = [
                cfg.num_slices_train,
                cfg.train_dim,
                cfg.train_dim,
                cfg.num_classes_seg,
            ]

            # set sample numbers
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

        cfg.batch_capacity_train = (
            4 * cfg.samples_per_volume
        )  # chosen as multiple of samples per volume

    def estimate_batch_size(self):
        """The batch size estimation is basically trail and error. So far tested
        with 128x128x2 patches in 2D and 128x128x32x2 in 3D, if using different
        values, guesstimate the relation to the memory.

        Returns
        -------
        int
            The recommended batch size
        """

        # TODO: this should be moved into the individual models.

        # set batch size
        # determine GPU memory (in MB)
        if len(tf.test.gpu_device_name()) > 0:
            gpu_number = int(tf.test.gpu_device_name()[-1])
            gpu_memory = int(np.round(GPUtil.getGPUs()[gpu_number].memoryTotal))
        else:
            # if no GPU was found, use 8 GB
            gpu_memory = 1024 * 1024 * 8

        a_name = self.hyper_parameters["architecture"].get_name()
        dim = self.hyper_parameters["dimensions"]

        if a_name == "UNet":
            # filters scale after the first filter, so use that for estimation
            first_f = self.hyper_parameters["network_parameters"]["n_filters"][0]
            if dim == 2:
                # this was determined by trail and error for 128x128x2 patches
                memory_consumption_guess = 4 * first_f
            elif dim == 3:
                # this was determined by trail and error for 128x128x32x2 patches
                memory_consumption_guess = 128 * first_f
            if "attention" in self.hyper_parameters["network_parameters"]:
                if (
                    self.hyper_parameters["network_parameters"]["attention"]
                    or self.hyper_parameters["network_parameters"]["encoder_attention"]
                    is not None
                ):
                    memory_consumption_guess *= 2
        elif a_name == "DenseTiramisu":
            if dim == 2:
                memory_consumption_guess = 512
            if dim == 3:
                memory_consumption_guess = 4096
        elif a_name == "DeepLabv3plus":
            backbone = self.hyper_parameters["network_parameters"]["backbone"]
            if dim == 2:
                if backbone == "resnet50":
                    memory_consumption_guess = 1024
                elif backbone == "resnet101":
                    memory_consumption_guess = 1024
                elif backbone == "resnet152":
                    memory_consumption_guess = 1024
                elif backbone == "mobilenet_v2":
                    memory_consumption_guess = 512
                elif backbone == "densenet121":
                    memory_consumption_guess = 1024
                elif backbone == "densenet169":
                    memory_consumption_guess = 1024
                elif backbone == "densenet201":
                    memory_consumption_guess = 1024
                elif backbone == "efficientnetB0":
                    memory_consumption_guess = 1024
                else:
                    raise NotImplementedError(
                        f"No heuristic implemented for backbone {backbone}."
                    )
            if dim == 3:
                raise NotImplementedError("3D Deeplab not implemented")
        else:
            raise NotImplementedError("No heuristic implemented for this network.")

        # return estimated recommended batch number
        return int(np.ceil(gpu_memory / memory_consumption_guess))

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
            file_dict=self.data_set,
            frac_obj=self.hyper_parameters["train_parameters"]["percent_of_object_samples"],
        )(
            train_files,
            batch_size=cfg.batch_size_train,
            n_epochs=self.hyper_parameters["train_parameters"]["epochs"],
            read_threads=cfg.train_reader_instances,
        )
        validation_dataset = SegLoader(
            mode=SegLoader.MODES.VALIDATE,
            name="validation_loader",
            file_dict=self.data_set,
            frac_obj=self.hyper_parameters["train_parameters"]["percent_of_object_samples"],
        )(
            vald_files,
            batch_size=cfg.batch_size_valid,
            read_threads=cfg.vald_reader_instances,
            n_epochs=self.hyper_parameters["train_parameters"]["epochs"],
        )

        # just use one sample with the foreground class using the validation files
        if self.tensorboard_images:
            visualization_dataset = SegLoader(
                name="visualization",
                file_dict=self.data_set,
                frac_obj=1,
                samples_per_volume=5,
                shuffle=False,
            )(
                vald_files,
                batch_size=cfg.batch_size_train,
                read_threads=1,
                n_epochs=self.hyper_parameters["train_parameters"]["epochs"],
            )
        else:
            visualization_dataset = None

        # only do a graph for the first fold
        write_graph = folder_name == "fold-0"

        net = self.hyper_parameters["architecture"](
            self.hyper_parameters["loss"],
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

        testloader = ApplyLoader(name="test_loader", file_dict=self.data_set)

        net = self.hyper_parameters["architecture"](
            self.hyper_parameters["loss"],
            is_training=False,
            model_path=str(self.output_path / folder_name / "models" / f"model-{version}"),
            **(self.hyper_parameters["network_parameters"]),
        )

        logger.info("Started applying %s to test datset.", folder_name)

        apply_path = self.output_path / folder_name / apply_name
        for file in tqdm(
            test_files, desc=f'{folder_name} ({apply_name.replace("_", " ")})', unit="file"
        ):
            f_name = Path(file).name

            # do inference
            result_image = apply_path / f"prediction-{f_name}-{version}{cfg.file_suffix}"
            if not result_image.exists():
                net.apply(testloader, file, apply_path=apply_path)

            # postprocess the image
            postprocessed_image = (
                apply_path / f"prediction-{f_name}-{version}-postprocessed{cfg.file_suffix}"
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

        eval_file_path = (
            self.output_path
            / folder_name
            / f"evaluation-{folder_name}-{version}_{name}.csv"
        )

        # remember the results
        results = []

        for file in test_files:
            prediction_path = apply_path / f"prediction-{file}-{version}{cfg.file_suffix}"
            if not "labels" in self.data_set[file]:
                logger.info("No labels found for %s", file)
                continue
            label_path = self.experiment_dir / self.data_set[file]["labels"]
            if not label_path.exists():
                logger.info("Label %s does not exists. It will be skipped", label_path)
                continue
            try:
                result_metrics = {"File Number": file}

                result_metrics = evaluation.evaluate_segmentation_prediction(
                    result_metrics, str(prediction_path), str(label_path)
                )

                # append result to eval file
                results.append(result_metrics)
                logger.info("        Finished Evaluation for %s", file)
            except RuntimeError as err:
                logger.exception(
                    "    !!! Evaluation of %s failed for %s, %s",
                    folder_name,
                    file,
                    err,
                )

        # write evaluation results
        if len(results) == 0:
            return
        results = pd.DataFrame(results)
        results.set_index("File Number", inplace=True)
        results.to_csv(eval_file_path, sep=";")

    def run_all_folds(self):
        """This is just a wrapper for run_fold and runs it for all folds"""
        self.hyper_parameters["evaluate_on_finetuned"] = False
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
            # set eval files
            eval_files = []
            for f_name in self.fold_dir_names:
                eval_files.append(
                    self.output_path / f_name / f"evaluation-{f_name}-{version}_{name}.csv"
                )
            if not np.all([f.exists() for f in eval_files]):
                raise FileNotFoundError("Eval file not found")
            # combine previous evaluations
            output_path = self.output_path / f"results_{name}_{version}"
            if not output_path.exists():
                output_path.mkdir()
            evaluation.combine_evaluation_results_from_folds(output_path, eval_files)
            # make plots
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
        # do not overwrite it
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
        }
        if hasattr(self, "external_test_set"):
            if self.external_test_set is not None:
                ext_set = [str(f) for f in self.external_test_set]
                experiment_dict["external_test_set"] = ext_set
        with open(self.experiment_file, "w") as f:
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
        with open(file, "r") as f:
            parameters = yaml.load(f, Loader=yaml.Loader)
        return cls(**parameters)

    def export_slurm_file(self, working_dir) -> Path:
        """Export the slurm files needed to start the training on the cluster.

        Parameters
        ----------
        working_dir : PathLike
            The current working directory

        Returns
        -------
        Path
            The location of the slurm file
        """
        run_script = Path(sys.argv[0]).resolve().parent / "run_single_experiment.py"
        job_dir = Path(os.environ["experiment_dir"]) / "slurm_jobs"
        if not job_dir.exists():
            job_dir.mkdir()
        log_dir = self.output_path / "slurm_logs"
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        job_file = job_dir / f"run_{self.name}.sh"
        if self.hyper_parameters["dimensions"] == 3:
            gpu_type = "GPU_no_K80"
        else:
            gpu_type = "GPU"
        export_slurm_job(
            filename=job_file,
            command=f"python {run_script} -f $SLURM_ARRAY_TASK_ID -e {self.output_path}",
            job_name=self.name,
            venv_dir=Path(sys.argv[0]).resolve().parent / "venv",
            workingdir=working_dir,
            job_type=gpu_type,
            hours=24,
            minutes=0,
            log_dir=log_dir,
            array_job=True,
            array_range=f"0-{self.folds-1}",
            variables={
                "data_dir": os.environ["data_dir"],
                "experiment_dir": os.environ["experiment_dir"],
            },
        )
        return job_file
