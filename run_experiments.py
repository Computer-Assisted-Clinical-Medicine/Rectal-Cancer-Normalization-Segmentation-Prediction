import os
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf

import evaluation
import SegmentationNetworkBasis.NetworkBasis.image as image
from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.architecture import DVN, CombiNet, UNet, VNet
from SegmentationNetworkBasis.NetworkBasis.util import (make_csv_file,
                                                        write_configurations,
                                                        write_metrics_to_csv)
from seg_data_loader import SegLoader, SegRatioLoader


class Experiment():

    def __init__(self, name, hyper_parameters, seed=42, output_path=None):
        self.hyper_parameters = hyper_parameters
        self.seed = seed
        self.name = name
        #set path different on the Server
        if cfg.ONSERVER:
            self.output_path = os.path.join("tmp", self.name)
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            if output_path == None:
                self.output_path = os.path.join('Experiments', self.name)
            else:
                self.output_path = output_path


    def set_seed(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _generate_folder_name(self):
        epochs = cfg.training_epochs // 10

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
            str(self.seed)
        ])

        return folder_name


    def _set_parameters_according_to_dimension(self):
        if self.hyper_parameters['dimensions'] == 2:
            cfg.num_channels = 3
            cfg.train_dim = 256
            cfg.samples_per_volume = 160
            cfg.batch_capacity_train = 750
            cfg.batch_capacity_valid = 450
            cfg.train_input_shape = [cfg.train_dim, cfg.train_dim, cfg.num_channels]
            cfg.train_label_shape = [cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
            print('   Train Shapes: ', cfg.train_input_shape, cfg.train_label_shape)
            cfg.test_dim = 512
            cfg.test_data_shape = [cfg.test_dim, cfg.test_dim, cfg.num_channels]
            cfg.test_label_shape = [cfg.test_dim, cfg.test_dim, cfg.num_classes_seg]
            print('   Test Shapes: ', cfg.test_data_shape, cfg.test_label_shape)
            cfg.batch_size_train = 16
            cfg.batch_size_test = 1
        elif self.hyper_parameters['dimensions'] == 3:
            cfg.num_channels = 1
            cfg.train_dim = 128
            cfg.samples_per_volume = 80
            cfg.batch_capacity_train = 250
            cfg.batch_capacity_valid = 150
            cfg.num_slices_train = 32
            cfg.train_input_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_channels]
            cfg.train_label_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
            print('   Train Shapes: ', cfg.train_input_shape, cfg.train_label_shape)
            cfg.test_dim = 512
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
        train_files = pd.read_csv(cfg.train_csv, dtype=object).values
        vald_files = pd.read_csv(cfg.vald_csv, dtype=object).values

        training_dataset = SegRatioLoader(name='training_loader') \
            (train_files, batch_size=cfg.batch_size_train, n_epochs=cfg.training_epochs,
            read_threads=cfg.train_reader_instances)
        validation_dataset = SegRatioLoader(mode=SegRatioLoader.MODES.VALIDATE, name='validation_loader') \
            (vald_files, batch_size=cfg.batch_size_train,
            read_threads=cfg.vald_reader_instances)

        net = self.hyper_parameters['architecture'](
            self.hyper_parameters['loss'],
            **(self.hyper_parameters["init_parameters"])
        )
        folder_name = self._generate_folder_name()
        write_configurations(self.hyper_parameters['experiment_path'], folder_name, net, cfg)
        # Train the network with the dataset iterators
        net.train(
            self.hyper_parameters['experiment_path'],
            folder_name,
            training_dataset,
            validation_dataset,
            summary_steps_per_epoch=cfg.summary_steps_per_epoch,
            **(self.hyper_parameters["train_parameters"])
        )


    def applying(self):
        '''!
        do testing

        '''
        tf.keras.backend.clear_session()

        # inits
        self.set_seed()
        test_files = pd.read_csv(cfg.test_csv, dtype=object).values
        testloader = SegLoader(mode=SegLoader.MODES.APPLY, name='test_loader')

        if self.hyper_parameters["evaluate_on_finetuned"]:
            cfg.training_epochs = cfg.epochs_for_finetuning
            original_folder = "-".join(os.path.basename(model_path).split('-')[0:2])
            folder_name = original_folder + '-' + self._make_folder_name() + '-f'
        else:
            cfg.training_epochs = cfg.epochs_for_training
            folder_name = self._make_folder_name()
        net = self.hyper_parameters['architecture'](self.hyper_parameters['loss'], is_training=False, model_path=os.path.join(self.hyper_parameters['experiment_path'], folder_name), **(self.hyper_parameters["init_parameters"]))

        for f in test_files:
            test_dataset = testloader(f, batch_size=cfg.batch_size_test, read_threads=cfg.vald_reader_instances)
            net.apply(test_dataset, f)

        tf.keras.backend.clear_session()

    def evaluate(self, model_path=''):
        '''!
        do testing

        '''

        self.set_seed()
        test_files = pd.read_csv(cfg.test_csv, dtype=object).values
        if self.hyper_parameters["evaluate_on_finetuned"]:
            cfg.training_epochs = cfg.epochs_for_finetuning
            original_folder = "-".join(os.path.basename(model_path).split('-')[0:2])
            folder_name = original_folder + '-' + self._generate_folder_name() + '-f'
        else:
            cfg.training_epochs = cfg.epochs_for_training
            folder_name = self._generate_folder_name()

        test_path = os.path.join(self.hyper_parameters['experiment_path'], folder_name + '_test')
        apply_path = os.path.join(self.hyper_parameters['experiment_path'], folder_name + '_apply')

        version = str(cfg.training_epochs)

        if not os.path.exists(test_path):
            os.makedirs(test_path)

        eval_file_path = os.path.join(test_path, 'evaluation-' + folder_name + '-' + version + '.csv')
        header_row = evaluation.make_csv_header()
        make_csv_file(eval_file_path, header_row)

        for f in test_files:
            folder, file_number = os.path.split(f[0])
            prediction_path = os.path.join(apply_path, ('prediction' + '-' + version + '-' + file_number + '.nii'))

            label_path = os.path.join(folder, (cfg.label_file_name_prefix + file_number + '.nii'))
            try:
                result_metrics = {}
                result_metrics['File Number'] = file_number

                result_metrics = evaluation.evaluate_segmentation_prediction(result_metrics, prediction_path, label_path)

                write_metrics_to_csv(eval_file_path, header_row, result_metrics)
                print('        Finished Evaluation for ', file_number)
            except RuntimeError as err:
                print("    !!! Evaluation of " + folder_name + ' failed for' + f[0], err)



    def fuse_probabilities_over_networks(self, dimensions_and_architectures):
        '''!
        do testing

        '''

        self.set_seed()
        np.seterr(under='ignore')
        test_files = pd.read_csv(cfg.test_csv, dtype=object).as_matrix()
        cfg.training_epochs = cfg.epochs_for_training
        for file in test_files:
            predictions = []
            weights = []
            for d, a in dimensions_and_architectures:
                self.hyper_parameters["dimensions"] = d
                self.hyper_parameters['architecture'] = a
                if a.get_name() == 'UNet':
                # if d == 2:
                    weights.append(0.3)
                else:
                    weights.append(0.2)
                # weights.append(0.25)
                folder_name = self._generate_folder_name()
                latest_model = tf.train.latest_checkpoint(os.path.join(self.hyper_parameters['experiment_path'], folder_name, 'model'))
                _, model_base = os.path.split(latest_model)
                name, ext = os.path.splitext(model_base)
                version = name.split('-')[-1]

                apply_path = os.path.join(self.hyper_parameters['experiment_path'], folder_name + '_apply')

                folder, file_number = os.path.split(file[0])
                prediction_path = os.path.join(apply_path, ('prediction' + '-' + version + '-' + file_number + '.nii'))
                predictions.append(sitk.ReadImage(prediction_path))

            self.hyper_parameters['architecture'] = CombiNet
            self.hyper_parameters["dimensions"] = 'n'
            folder_name = self._generate_folder_name()
            apply_path = os.path.join(self.hyper_parameters['experiment_path'], folder_name + '_apply')
            if not os.path.exists(apply_path):
                os.makedirs(apply_path)
            fused_probabilities = weights[0] * sitk.GetArrayFromImage(predictions[0])
            for w, p in zip(weights[1:], predictions[1:]):
                fused_probabilities = fused_probabilities + w * sitk.GetArrayFromImage(p)

            data_info = image.get_data_info(p)
            fused_prediction = np.argmax(fused_probabilities, -1)
            pred_img = image.np_array_to_itk_image(fused_prediction, data_info, cfg.label_background_value,
                                                cfg.adapt_resolution, cfg.target_type_label)
            sitk.WriteImage(pred_img, os.path.join(apply_path,
                                    ('prediction' + '-' + version + '-' + file_number + '.nii')))
            print('        Fused predictions for ', file_number)


    def run(self, data, k_fold, dimensions_and_architectures, losses):
        self.hyper_parameters["evaluate_on_finetuned"] = False
        # Experiment 1: Train on individual Data sets
        for (data_name, data_set) in data:
            self.set_seed()

            self.hyper_parameters['experiment_path'] = os.path.join(self.output_path, 'individual_final5f-fin_' + data_name)
            all_indices = np.random.permutation(range(0, data_set.size))
            test_folds = np.array_split(all_indices, k_fold)

            for f in range(0, k_fold):
                test_indices = test_folds[f]
                remaining_indices = np.setdiff1d(all_indices, test_folds[f])
                vald_indices = remaining_indices[:cfg.number_of_vald]
                train_indices = remaining_indices[cfg.number_of_vald:]

                train_files = data_set[train_indices]
                vald_files = data_set[vald_indices]
                test_files = data_set[test_indices]

                np.savetxt(cfg.train_csv, train_files, fmt='%s', header='path')
                np.savetxt(cfg.vald_csv, vald_files, fmt='%s', header='path')
                np.savetxt(cfg.test_csv, test_files, fmt='%s', header='path')

                cfg.num_files = len(train_files)

                print('  Data Set ' + data_name + str(f) + ': ' + str(train_indices.size) + ' train cases, '
                    + str(test_indices.size)
                    + ' test cases, ' + str(vald_indices.size) + ' vald cases')

                cfg.training_epochs = cfg.epochs_for_training

                for d, a in dimensions_and_architectures:
                    self.hyper_parameters["dimensions"] = d
                    self._set_parameters_according_to_dimension()
                    self.hyper_parameters['architecture'] = a
                    for l in losses:
                        self.hyper_parameters["loss"] = l

                        try:
                            cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL
                            cfg.percent_of_object_samples = 50
                            self.training()
                            pass
                        except Exception as err:
                            print('Training ' + data_name,
                                  self.hyper_parameters['architecture'].get_name() + self.hyper_parameters['loss'] + 'failed!')
                            print(err)
                        
                        try:
                            cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
                            self.applying()
                        except Exception as err:
                            print('Applying ' + data_name,
                                  self.hyper_parameters['architecture'].get_name() + self.hyper_parameters['loss'] + 'failed!')
                            print(err)
                        
                        try:
                            self.evaluate()
                        except Exception as err:
                            print('Evaluating ' + data_name,
                                self.hyper_parameters['architecture'].get_name() + self.hyper_parameters['loss'] + 'failed!')
                            print(err)

            evaluation.combine_evaluation_results_from_folds(self.hyper_parameters['experiment_path'],
                                                            losses, dimensions_and_architectures)
            evaluation.make_boxplot_graphic(self.hyper_parameters['experiment_path'], dimensions_and_architectures, losses)


if __name__ == '__main__':

    data_file = 'data.csv'
    data_list = pd.read_csv(data_file, dtype=object, header=None).values

    k_fold = 5
    losses = ['CEL+DICE']
    dimensions_and_architectures = ([2, UNet], [2, VNet], [3, UNet],[3, VNet])

    init_parameters = {
        "regularize": [True, 'L2', 0.0000001],
        "drop_out": [True, 0.01],
        "activation": "elu",
        "do_batch_normalization": False,
        "do_bias": True,
        "cross_hair": False
    }

    train_parameters = {
        "l_r": 0.001,
        "optimizer": "Adam"
    }

    hyper_parameters = {
        "init_parameters": init_parameters,
        "train_parameters": train_parameters
    }

    cfg.epochs_for_training = 80

    experiment = Experiment(
        hyper_parameters=hyper_parameters,
        name='testExperiment'
    )

    experiment.run([('ircad', data_list)], k_fold, dimensions_and_architectures, losses)
