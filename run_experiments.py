import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf

import evaluation
import SegmentationNetworkBasis.NetworkBasis.image as image
from seg_data_loader import SegLoader, SegRatioLoader
from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.architecture import DVN, CombiNet, UNet, VNet
from SegmentationNetworkBasis.NetworkBasis.util import (make_csv_file,
                                                        write_configurations,
                                                        write_metrics_to_csv)


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

        #check for finetuning
        if not hasattr(self.hyper_parameters, 'evaluate_on_finetuned'):
            self.hyper_parameters["evaluate_on_finetuned"]=False

    def set_seed(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _generate_folder_name(self):
        epochs = self.hyper_parameters['train_parameters']['epochs']

        if self.hyper_parameters['init_parameters']['drop_out'][0]:
            do = 'DO'
        else:
            do = 'nDO'

        if self.hyper_parameters['init_parameters']['do_batch_normalization']:
            bn = 'BN'
        else:
            bn = 'nBN'

        folder_name = "-".join([
            self.hyper_parameters['architecture'].get_name() + str(self.hyper_parameters['dimensions']) + 'D',
            self.hyper_parameters['loss'],
            do,
            bn,
            str(epochs),
            str(self.seed),
            f'fold{self.fold}'
        ])

        return folder_name


    def _set_parameters_according_to_dimension(self):
        if self.hyper_parameters['dimensions'] == 2:
            cfg.num_channels = 3
            #cfg.train_dim = 256
            cfg.samples_per_volume = 160
            cfg.batch_capacity_train = 750
            cfg.batch_capacity_valid = 450
            cfg.train_input_shape = [cfg.train_dim, cfg.train_dim, cfg.num_channels]
            cfg.train_label_shape = [cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
            print('   Train Shapes: ', cfg.train_input_shape, cfg.train_label_shape)
            #cfg.test_dim = 512
            cfg.test_data_shape = [cfg.test_dim, cfg.test_dim, cfg.num_channels]
            cfg.test_label_shape = [cfg.test_dim, cfg.test_dim, cfg.num_classes_seg]
            print('   Test Shapes: ', cfg.test_data_shape, cfg.test_label_shape)
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
            print('   Train Shapes: ', cfg.train_input_shape, cfg.train_label_shape)
            #cfg.test_dim = 512
            cfg.num_slices_test = 32
            cfg.test_data_shape = [cfg.num_slices_test, cfg.test_dim, cfg.test_dim, cfg.num_channels]
            cfg.test_label_shape = [cfg.num_slices_test, cfg.test_dim, cfg.test_dim, cfg.num_classes_seg]
            print('   Test Shapes: ', cfg.test_data_shape, cfg.test_label_shape)
            cfg.batch_size_train = 8
            cfg.batch_size_test = 1


    def training(self):
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
            #add parameters needed from other sets
            #add initialization parameters
            **self.hyper_parameters["init_parameters"]
        )
        folder_name = self._generate_folder_name()
        write_configurations(self.output_path, folder_name, net, cfg)
        # Train the network with the dataset iterators
        net.train(
            self.output_path,
            folder_name,
            training_dataset,
            validation_dataset,
            summary_steps_per_epoch=cfg.summary_steps_per_epoch,
            #add training parameters
            **(self.hyper_parameters["train_parameters"])
        )


    def applying(self, model_path=None):
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

        if self.hyper_parameters["evaluate_on_finetuned"]:
            original_folder = "-".join(model_path.name.split('-')[0:2])
            folder_name = original_folder + '-' + self._make_folder_name() + '-f'
        else:
            folder_name = self._generate_folder_name()
        net = self.hyper_parameters['architecture'](
            self.hyper_parameters['loss'],
            is_training=False,
            model_path=str(self.output_path/folder_name),
            **(self.hyper_parameters["init_parameters"])
            )

        for f in self.test_files:
            test_dataset = testloader(f, batch_size=cfg.batch_size_test, read_threads=cfg.vald_reader_instances)
            net.apply(test_dataset, f)

        tf.keras.backend.clear_session()

    def evaluate(self, model_path=None):
        '''!
        do testing

        '''

        self.set_seed()
        if self.hyper_parameters["evaluate_on_finetuned"]:
            original_folder = "-".join(model_path.name.split('-')[0:2])
            folder_name = original_folder + '-' + self._generate_folder_name() + '-f'
        else:
            folder_name = self._generate_folder_name()

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

                write_metrics_to_csv(eval_file_path, header_row, result_metrics)
                print('        Finished Evaluation for ', file_number)
            except RuntimeError as err:
                print("    !!! Evaluation of " + folder_name + ' failed for' + f.name, err)
        
        #remember eval files
        self.eval_files.append(eval_file_path)


    def run(self, data_set, k_fold=1):
        self.hyper_parameters["evaluate_on_finetuned"] = False
        # Experiment 1: Train on individual Data sets
        self.set_seed()

        all_indices = np.random.permutation(range(0, data_set.size))
        #split the data into k_fold sections
        if k_fold > 0:
            test_folds = np.array_split(all_indices, k_fold)
        else:
            #otherwise, us cfg.data_train_split
            test_folds = all_indices[int(all_indices.size*cfg.data_train_split):].reshape(1,-1)

        #remember eval files for later use
        self.eval_files = []

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

            self.workingdir = self.output_path/self._generate_folder_name()

            if not self.workingdir.exists():
                self.workingdir.mkdir()

            np.savetxt(self.workingdir/cfg.train_csv, self.train_files, fmt='%s', header='path')
            np.savetxt(self.workingdir/cfg.vald_csv, self.vald_files, fmt='%s', header='path')
            np.savetxt(self.workingdir/cfg.test_csv, self.test_files, fmt='%s', header='path')

            cfg.num_files = len(self.train_files)

            print('  Data Set ' + str(f) + ': ' + str(train_indices.size) + ' train cases, '
                + str(test_indices.size)
                + ' test cases, ' + str(vald_indices.size) + ' vald cases')

            self._set_parameters_according_to_dimension()

            #try:
            cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL
            cfg.percent_of_object_samples = 50
            self.training()
            #    pass
            #except Exception as err:
            #    print('Training ',
            #            self.hyper_parameters['architecture'].get_name() + self.hyper_parameters['loss'] + 'failed!')
            #    print(err)
            
            try:
                cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
                self.applying()
            except Exception as err:
                print('Applying ',
                        self.hyper_parameters['architecture'].get_name() + self.hyper_parameters['loss'] + 'failed!')
                print(err)
            
            try:
                self.evaluate()
            except Exception as err:
                print('Evaluating ',
                    self.hyper_parameters['architecture'].get_name() + self.hyper_parameters['loss'] + 'failed!')
                print(err)

        evaluation.combine_evaluation_results_from_folds(
            self.output_path,
            self.eval_files
        )
        evaluation.make_boxplot_graphic(
            self.output_path,
            self.eval_files
        )


if __name__ == '__main__':

    data_dir = Path('TestData')
    data_list = np.array([str(d) for d in data_dir.iterdir() if d.is_dir()])

    k_fold = 2
    dimensions_and_architectures = ([2, UNet], [2, VNet], [3, UNet],[3, VNet])

    #define the parameters that are constant
    init_parameters = {
        "regularize": [True, 'L2', 0.0000001],
        "drop_out": [True, 0.01],
        "activation": "elu",
        "do_batch_normalization": False,
        "do_bias": True,
        "cross_hair": False,
        "do_gradient_clipping" : False,
        "clipping_value" : 50
    }

    train_parameters = {
        "l_r": 0.001,
        "optimizer": "Adam",
        "epochs" : 1 #TODO: increase
    }

    constant_parameters = {
        "init_parameters": init_parameters,
        "train_parameters": train_parameters,
        "loss" : 'DICE'
    }

    #generate a set of hyperparameters for each dimension and architecture
    for d, a in dimensions_and_architectures:
        hyper_parameters = {
            **constant_parameters,
            'dimensions' : d,
            'architecture' : a
        }

        #define experiment
        experiment_name = f'{a.get_name()}{d}D'

        experiment = Experiment(
            hyper_parameters=hyper_parameters,
            name=experiment_name,
            output_path=Path('Experiments', 'testExperiment', experiment_name)
        )

        experiment.run(data_list, k_fold)
