import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf
from tqdm import tqdm

import evaluation
import SegmentationNetworkBasis.NetworkBasis.image as image
from seg_data_loader import SegLoader, SegRatioLoader
from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.NetworkBasis.util import (make_csv_file,
                                                        write_configurations,
                                                        write_metrics_to_csv)

#configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Experiment():

    def __init__(self, name, hyper_parameters, seed=42, output_path=None):
        """Run experiments using a fixed set of hyperparameters

        Parameters
        ----------
            name : str
                Name of the experiment, is used for the folder name
            hyper_parameters : dict
                the hyperparameters that should be used
            seed : int, optional
                the global seed, by default 42
            output_path : str, optional
                path to write output in, if None and not on server, Experiments is used, by default None
        """
        self.hyper_parameters = hyper_parameters
        self.seed = seed
        self.name = name
        #start with fold 0
        self.fold = 0
        #set path different on the Server
        if cfg.ONSERVER:
            self.output_path = Path("tmp", self.name)
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            if output_path == None:
                self.output_path = Path('Experiments', self.name)
            else:
                self.output_path = Path(output_path)

        if not self.output_path.exists():
            self.output_path.mkdir()
        logger.info('Set %s as output folder, all output will be there', self.output_path)

        #check for finetuning
        if not hasattr(self.hyper_parameters, 'evaluate_on_finetuned'):
            self.hyper_parameters["evaluate_on_finetuned"]=False

        #set hyperparameterfile to store all hyperparameters
        self.hyperparameter_file = self.output_path / 'hyperparameters.json'

    def set_seed(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)


    def _set_parameters_according_to_dimension(self):
        if self.hyper_parameters['dimensions'] == 2:
            cfg.num_channels = 3
            #cfg.train_dim = 256
            cfg.samples_per_volume = 160
            cfg.batch_capacity_train = 750
            cfg.batch_capacity_valid = 450
            cfg.train_input_shape = [cfg.train_dim, cfg.train_dim, cfg.num_channels]
            cfg.train_label_shape = [cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
            logger.debug('   Train Shapes: %s (input), %s (labels)', cfg.train_input_shape, cfg.train_label_shape)
            #cfg.test_dim = 512
            cfg.test_data_shape = [cfg.test_dim, cfg.test_dim, cfg.num_channels]
            cfg.test_label_shape = [cfg.test_dim, cfg.test_dim, cfg.num_classes_seg]
            logger.debug('   Test Shapes: %s (input) %s (labels)', cfg.test_data_shape, cfg.test_label_shape)
            cfg.batch_size_train = 16
            cfg.batch_size_test = 1
        elif self.hyper_parameters['dimensions'] == 3:
            cfg.num_channels = 1
            #cfg.train_dim = 128
            cfg.samples_per_volume = 80
            cfg.batch_capacity_train = 250
            cfg.batch_capacity_valid = 150
            cfg.num_slices_train = 32
            cfg.train_input_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_channels]
            cfg.train_label_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
            logger.debug('   Train Shapes: %s (input), %s (labels)', cfg.train_input_shape, cfg.train_label_shape)
            #cfg.test_dim = 512
            cfg.num_slices_test = 32
            cfg.test_data_shape = [cfg.num_slices_test, cfg.test_dim, cfg.test_dim, cfg.num_channels]
            cfg.test_label_shape = [cfg.num_slices_test, cfg.test_dim, cfg.test_dim, cfg.num_classes_seg]
            logger.debug('   Test Shapes: %s (input) %s (labels)', cfg.test_data_shape, cfg.test_label_shape)
            cfg.batch_size_train = 8
            cfg.batch_size_test = 1


    def training(self, folder_name):
        tf.keras.backend.clear_session()
        # inits
        self.set_seed()

        #generate loader
        training_loader = SegRatioLoader(name='training_loader')
        training_dataset = training_loader(
            self.train_files,
            batch_size=cfg.batch_size_train,
            n_epochs=self.hyper_parameters['train_parameters']['epochs'],
            read_threads=cfg.train_reader_instances
        )
        validation_dataset = SegRatioLoader(
            mode=SegRatioLoader.MODES.VALIDATE,
            name='validation_loader'
        )(
            self.vald_files, batch_size=cfg.batch_size_train,
            read_threads=cfg.vald_reader_instances
        )

        net = self.hyper_parameters['architecture'](
            self.hyper_parameters['loss'],
            #add initialization parameters
            **self.hyper_parameters["init_parameters"]
        )
        write_configurations(self.output_path, folder_name, net, cfg)
        # Train the network with the dataset iterators
        logger.info('Started training of %s', folder_name)
        net.train(
            logs_path=str(self.output_path),
            folder_name=folder_name,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            summary_steps_per_epoch=cfg.summary_steps_per_epoch,
            #add training parameters
            **(self.hyper_parameters["train_parameters"])
        )


    def applying(self, folder_name, model_path=None):
        '''!
        do testing

        '''
        tf.keras.backend.clear_session()

        # inits
        self.set_seed()
        testloader = SegLoader(
            mode=SegLoader.MODES.APPLY,
            name='test_loader'
        )

        net = self.hyper_parameters['architecture'](
            self.hyper_parameters['loss'],
            is_training=False,
            model_path=str(self.output_path/folder_name),
            **(self.hyper_parameters["init_parameters"])
            )

        logger.info('Started applying %s to test datset.', folder_name)
        for f in tqdm(self.test_files, desc=f'{folder_name} (test)', unit='file', position=3):
            test_dataset = testloader(f, batch_size=cfg.batch_size_test, read_threads=cfg.vald_reader_instances)
            net.apply(test_dataset, f)

        tf.keras.backend.clear_session()

    def evaluate_fold(self, folder_name, model_path=None):
        '''!
        do testing

        '''

        self.set_seed()

        logger.info('Start evaluation of %s.', folder_name)

        apply_path = self.output_path/folder_name/'apply'
        if not apply_path.exists():
            apply_path.mkdir()

        version = str(self.hyper_parameters['train_parameters']['epochs'])

        eval_file_path = self.workingdir / f'evaluation-{folder_name}-{version}_test.csv'
        header_row = evaluation.make_csv_header()
        make_csv_file(eval_file_path, header_row)

        for f in self.test_files:
            f = Path(f)
            folder, file_number = f.parts
            prediction_path = Path(apply_path) / f'prediction-{f.name}-{version}.nii.gz'

            label_path = f/cfg.label_file_name
            try:
                result_metrics = {}
                result_metrics['File Number'] = file_number

                result_metrics = evaluation.evaluate_segmentation_prediction(result_metrics, str(prediction_path), str(label_path))

                #append result to eval file
                write_metrics_to_csv(eval_file_path, header_row, result_metrics)
                logger.info('        Finished Evaluation for %s', file_number)
            except RuntimeError as err:
                logger.error("    !!! Evaluation of %s failed for %s, %s", folder_name, f.name, err)
        
        #remember eval files
        self.eval_files.append(eval_file_path)


    def run(self, data_set, k_fold=1):
        self.hyper_parameters["evaluate_on_finetuned"] = False
        self.set_seed()

        all_indices = np.random.permutation(range(0, data_set.size))
        #split the data into k_fold sections
        if k_fold > 1:
            test_folds = np.array_split(all_indices, k_fold)
        else:
            #otherwise, us cfg.data_train_split
            test_folds = all_indices[int(all_indices.size*cfg.data_train_split):].reshape(1,-1)

        #remember eval files for later use
        self.eval_files = []

        #export parameters
        self.export_hyperparameters()

        for f in range(0, k_fold):
            self.fold = f

            #test is the section
            test_indices = test_folds[f]
            remaining_indices = np.setdiff1d(all_indices, test_folds[f])
            #number of validation is set in config
            vald_indices = remaining_indices[:cfg.number_of_vald]
            #the rest is used for training
            train_indices = remaining_indices[cfg.number_of_vald:]

            self.train_files = data_set[train_indices]
            self.vald_files = data_set[vald_indices]
            self.test_files = data_set[test_indices]

            folder_name = f'fold-{f}'
            self.workingdir = self.output_path/folder_name
            logger.info('workingdir is %s', self.workingdir)

            if not self.workingdir.exists():
                self.workingdir.mkdir()

            np.savetxt(self.workingdir/cfg.train_csv, self.train_files, fmt='%s', header='path')
            np.savetxt(self.workingdir/cfg.vald_csv, self.vald_files, fmt='%s', header='path')
            np.savetxt(self.workingdir/cfg.test_csv, self.test_files, fmt='%s', header='path')

            cfg.num_files = len(self.train_files)

            logger.info(
                '  Data Set %s: %s  train cases, %s  test cases, %s vald cases',
                f, train_indices.size, test_indices.size, vald_indices.size
            )

            self._set_parameters_according_to_dimension()

            tqdm.write(f'Starting with {self.name} {folder_name} (Fold {f} of {k_fold})')

            #try the actual training
            try:
                cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL
                cfg.percent_of_object_samples = 50
                self.training(folder_name=folder_name)
            except tf.errors.ResourceExhaustedError as err:
                logger.error(
                    'The model did not fit in memory, Training %s %s failed!',
                    self.hyper_parameters['architecture'].get_name(), self.hyper_parameters['loss']
                )
                logger.error(err)
                break 
            except Exception as err:
                logger.error(
                    'Training %s %s failed!',
                    self.hyper_parameters['architecture'].get_name(), self.hyper_parameters['loss']
                    )
                logger.error(err)
                break
            
            try:
                cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
                self.applying(folder_name=folder_name)
            except Exception as err:
                logger.error('Applying %s %s failed!',
                        self.hyper_parameters['architecture'].get_name(), self.hyper_parameters['loss'])
                logger.error(err)
            
            try:
                self.evaluate_fold(folder_name=folder_name)
            except Exception as err:
                logger.error('Evaluating %s %s failed!',
                    self.hyper_parameters['architecture'].get_name(), self.hyper_parameters['loss'])
                logger.error(err)

            tqdm.write(f'Finished with {self.name} {folder_name} (Fold {f} of {k_fold})')



    def evaluate(self):
        #combine previous evaluations
        evaluation.combine_evaluation_results_from_folds(
            self.output_path,
            self.eval_files
        )
        #make plots
        evaluation.make_boxplot_graphic(
            self.output_path,
            self.eval_files
        )

    def export_hyperparameters(self):
        params = self.hyper_parameters.copy()
        if 'architecture' in params:
            params['architecture'] = params['architecture'].get_name()

        with open(self.hyperparameter_file, 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)