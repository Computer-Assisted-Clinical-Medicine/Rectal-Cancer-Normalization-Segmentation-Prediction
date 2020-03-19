import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.NetworkBasis.util import write_configurations, write_metrics_to_csv, make_csv_file
from SegmentationNetworkBasis.architecture import ShallowVNet, UNet, VNet, DVN
from vesselsegloader import VesselSegRatioLoader
from vesselsegloader import VesselSegLoader
import evaluation

experiment_name = "vessel_segmentation"
if cfg.ONSERVER:
    logs_path = os.path.join("tmp", experiment_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    logs_path = os.path.join("R:", experiment_name)


def _make_folder_name(hyper_parameters, seed):
    epochs = cfg.training_epochs // 10

    if hyper_parameters['init_parameters']['drop_out'][0]:
        do = 'DO'
    else:
        do = 'nDO'

    if hyper_parameters['init_parameters']['do_batch_normalization']:
        bn = 'BN'
    else:
        bn = 'nBN'

    folder_name = "-".join([hyper_parameters['architecture'].get_name() + str(hyper_parameters['dimensions']) + 'D',
                            hyper_parameters['loss'], do, bn, str(epochs), str(seed)])

    return folder_name


def _set_parameters_according_to_dimension(hyper_parameters):
    if hyper_parameters['dimensions'] == 2:
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
    elif hyper_parameters['dimensions'] == 3:
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


def training(seed=42, **hyper_parameters):
    tf.keras.backend.clear_session()
    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)
    train_files = pd.read_csv(cfg.train_csv, dtype=object).as_matrix()
    vald_files = pd.read_csv(cfg.vald_csv, dtype=object).as_matrix()

    training_dataset = VesselSegRatioLoader(name='training_loader') \
        (train_files, batch_size=cfg.batch_size_train, n_epochs=cfg.training_epochs,
         read_threads=cfg.train_reader_instances)
    validation_dataset = VesselSegRatioLoader(mode=VesselSegRatioLoader.MODES.VALIDATE, name='validation_loader') \
        (vald_files, batch_size=cfg.batch_size_train,
         read_threads=cfg.vald_reader_instances)

    net = hyper_parameters['architecture'](hyper_parameters['loss'], **(hyper_parameters["init_parameters"]))
    folder_name = _make_folder_name(hyper_parameters, seed)
    write_configurations(hyper_parameters['experiment_path'], folder_name, net, cfg)
    # Train the network with the dataset iterators
    net.train(hyper_parameters['experiment_path'], folder_name, training_dataset, validation_dataset,
              summary_steps_per_epoch=cfg.summary_steps_per_epoch, **(hyper_parameters["train_parameters"]))


def finetuning(seed=42, **hyper_parameters):
    tf.keras.backend.clear_session()
    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)
    fine_files = pd.read_csv(cfg.fine_csv, dtype=object).as_matrix()
    vald_files = pd.read_csv(cfg.vald_csv, dtype=object).as_matrix()

    cfg.training_epochs = 80

    training_dataset = VesselSegRatioLoader(name='training_loader') \
        (fine_files, batch_size=cfg.batch_size_train, n_epochs=cfg.training_epochs,
         read_threads=cfg.train_reader_instances)
    validation_dataset = VesselSegRatioLoader(mode=VesselSegRatioLoader.MODES.VALIDATE, name='validation_loader') \
        (vald_files, batch_size=cfg.batch_size_train,
         read_threads=cfg.vald_reader_instances)

    cfg.training_epochs = 20
    folder_name = _make_folder_name(hyper_parameters, 0)

    cfg.training_epochs = 80

    net = hyper_parameters['architecture'](hyper_parameters['loss'], do_finetune=True,
                                           model_path=os.path.join(hyper_parameters['experiment_path'], folder_name),
                                           **(hyper_parameters["init_parameters"]))

    folder_name = _make_folder_name(hyper_parameters, seed)
    write_configurations(hyper_parameters['experiment_path'], folder_name, net, cfg)
    # Train the network with the dataset iterators
    net.finetune(hyper_parameters['experiment_path'], folder_name, training_dataset, validation_dataset,
              summary_steps_per_epoch=cfg.summary_steps_per_epoch, **(hyper_parameters["train_parameters"]))


def applying(seed=42, **hyper_parameters):
    '''!
    do testing

    '''
    tf.keras.backend.clear_session()

    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)
    test_files = pd.read_csv(cfg.test_csv, dtype=object).as_matrix()
    testloader = VesselSegLoader(mode=VesselSegLoader.MODES.APPLY, name='test_loader')

    folder_name = _make_folder_name(hyper_parameters, seed)
    if hyper_parameters["evaluate_on_finetuned"]:
        folder_name = folder_name + '-f'
    net = hyper_parameters['architecture'](hyper_parameters['loss'], is_training=False, model_path=os.path.join(hyper_parameters['experiment_path'], folder_name), **(hyper_parameters["init_parameters"]))

    for f in test_files:
        test_dataset = testloader(f, batch_size=cfg.batch_size_test, read_threads=cfg.vald_reader_instances)
        net.apply(test_dataset, f)

    tf.keras.backend.clear_session()


def evaluate(seed=42, **hyper_parameters):
    '''!
    do testing

    '''

    np.random.seed(seed)
    test_files = pd.read_csv(cfg.test_csv, dtype=object).as_matrix()
    folder_name = _make_folder_name(hyper_parameters, seed)
    if hyper_parameters["evaluate_on_finetuned"]:
        folder_name = folder_name + '-f'
    test_path = os.path.join(hyper_parameters['experiment_path'], folder_name + '_test')
    apply_path = os.path.join(hyper_parameters['experiment_path'], folder_name + '_apply')
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    latest_model = tf.train.latest_checkpoint(os.path.join(hyper_parameters['experiment_path'], folder_name, 'model'))
    _, model_base = os.path.split(latest_model)
    file, ext = os.path.splitext(model_base)
    version = file.split('-')[-1]
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


def experiment_1(data, hyper_parameters, k_fold, dimensions, losses, architectures):
    hyper_parameters["evaluate_on_finetuned"] = False
    # Experiment 1: Train on individual Data sets
    for (data_name, data_set) in data:
        np.random.seed(42)

        hyper_parameters['experiment_path'] = os.path.join(logs_path, 'individual_final5f_' + data_name)
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

            cfg.training_epochs = 80

            for d in dimensions:
                hyper_parameters["dimensions"] = d
                _set_parameters_according_to_dimension(hyper_parameters)
                for a in architectures:
                    hyper_parameters['architecture'] = a
                    for l in losses:
                        hyper_parameters["loss"] = l

                        try:
                            cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL
                            cfg.percent_of_object_samples = 50
                            training(**hyper_parameters, seed=f)
                            pass
                        except Exception as err:
                            print('Training ' + data_name,
                                  hyper_parameters['architecture'].get_name() + hyper_parameters['loss'] + 'failed!')
                            print(err)

                        try:
                            cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
                            applying(**hyper_parameters, seed=f)
                        except Exception as err:
                            print('Applying ' + data_name,
                                  hyper_parameters['architecture'].get_name() + hyper_parameters['loss'] + 'failed!')
                            print(err)

                        try:
                            evaluate(**hyper_parameters, seed=f)
                        except Exception as err:
                            print('Evaluating ' + data_name,
                                  hyper_parameters['architecture'].get_name() + hyper_parameters['loss'] + 'failed!')
                            print(err)

        try:
            evaluation.combine_evaluation_results_from_folds(hyper_parameters['experiment_path'],
                                                             k_fold, dimensions, losses, architectures)
            pass
        except Exception as err:
            print('Could not combine results!')
            print(err)

        evaluation.make_boxplot_graphic(hyper_parameters['experiment_path'], dimensions, losses, architectures)


def experiment_2(data_train, data_fine, hyper_parameters, k_fold, dimensions, losses, architectures):
    hyper_parameters["evaluate_on_finetuned"] = True
    # Experiment 2: Pretrain on synthetic, Finetune on Real
    for (fine_data_name, fine_data_set) in data_fine:
        np.random.seed(42)
        all_indices = np.random.permutation(range(0, fine_data_set.size))
        test_folds = np.array_split(all_indices, k_fold)

        for (train_data_name, train_data_set) in data_train:
            np.random.seed(42)

            train_indices = np.random.permutation(range(0, train_data_set.size))
            train_files = train_data_set[train_indices]
            np.savetxt(cfg.train_csv, train_files, fmt='%s', header='path')

            hyper_parameters['experiment_path'] = os.path.join(logs_path,
                                                   'trainandfinetune_' + train_data_name + '_' + fine_data_name)

            for f in range(k_fold):
                test_indices = test_folds[f]
                remaining_indices = np.setdiff1d(all_indices, test_folds[f])
                vald_indices = remaining_indices[:cfg.number_of_vald]
                fine_indices = remaining_indices[cfg.number_of_vald:]

                fine_files = fine_data_set[fine_indices]
                vald_files = fine_data_set[vald_indices]
                test_files = fine_data_set[test_indices]

                np.savetxt(cfg.fine_csv, fine_files, fmt='%s', header='path')
                np.savetxt(cfg.vald_csv, vald_files, fmt='%s', header='path')
                np.savetxt(cfg.test_csv, test_files, fmt='%s', header='path')

                print('  Data Set ' + train_data_name + '_' + fine_data_name + str(f) + ': '
                      + str(train_indices.size) + ' train cases, ' + str(fine_indices.size) + ' fine cases, '
                      + str(test_indices.size) + ' test cases, ' + str(vald_indices.size) + ' vald cases')

                for d in dimensions:
                    hyper_parameters["dimensions"] = d
                    _set_parameters_according_to_dimension(hyper_parameters)
                    for l in losses:
                        hyper_parameters["loss"] = l
                        for a in architectures:
                            hyper_parameters['architecture'] = a

                            if f == 0:
                                try:
                                    # cfg.num_files = len(train_files)
                                    # hyper_parameters['train_parameters']['l_r'] = 0.001
                                    # cfg.training_epochs = 20
                                    # cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL
                                    # cfg.percent_of_object_samples = 50
                                    # training(**hyper_parameters, seed=f)
                                    pass
                                except Exception as err:
                                    print('Training ' + train_data_name,
                                          hyper_parameters['architecture'].get_name() + hyper_parameters[
                                              'loss'] + 'failed!')
                                    print(err)

                            # fine tuning epochs are set in fine tune function

                            try:
                                cfg.num_files = len(fine_files)
                                hyper_parameters['train_parameters']['l_r'] = 0.0001
                                cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL
                                cfg.percent_of_object_samples = 50
                                finetuning(**hyper_parameters, seed=f)
                                pass
                            except Exception as err:
                                print('Training ' + train_data_name,
                                      hyper_parameters['architecture'].get_name() + hyper_parameters[
                                          'loss'] + 'failed!')
                                print(err)

                            # try:
                            #     cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
                            #     applying(**hyper_parameters, seed=f)
                            # except Exception as err:
                            #     print('Applying ' + fine_data_name,
                            #           hyper_parameters['architecture'].get_name() + hyper_parameters[
                            #               'loss'] + 'failed!')
                            #     print(err)
                            #
                            # try:
                            #     evaluate(**hyper_parameters, seed=f)
                            # except Exception as err:
                            #     print('Evaluating ' + fine_data_name,
                            #           hyper_parameters['architecture'].get_name() + hyper_parameters[
                            #               'loss'] + 'failed!')
                            #     print(err)

        try:
            evaluation.combine_evaluation_results_from_folds(
                hyper_parameters['experiment_path'],
                k_fold, dimensions, losses, architectures)
            evaluation.make_boxplot_graphic(hyper_parameters['experiment_path'], dimensions, losses, architectures)
        except Exception as err:
            print('Could not combine results for fold ' + str(f) + ' failed!')
            print(err)


if __name__ == '__main__':

    np.random.seed(42)

    ircad_csv = 'ircad.csv'
    all_ircad_files = pd.read_csv(ircad_csv, dtype=object).as_matrix()
    # btcv_csv = 'btcv.csv'
    # all_btcv_files = pd.read_csv(btcv_csv, dtype=object).as_matrix()
    # synth_csv = 'synth.csv'
    # all_synth_files = pd.read_csv(synth_csv, dtype=object).as_matrix()
    xcat_csv = 'xcat.csv'
    all_xcat_files = pd.read_csv(xcat_csv, dtype=object).as_matrix()
    gan_csv = 'gan.csv'
    all_gan_files = pd.read_csv(gan_csv, dtype=object).as_matrix()

    k_fold = 5
    losses = ['DICE', 'CEL']  #'WCEL', 'NCEL', 'ECEL', 'WDL',
    dimensions = [2]
    architectures = [UNet, VNet]  #, FCN

    init_parameters = {"regularize": [True, 'L2', 0.0000001], "activation": "relu", "drop_out": [True, 0.01],
                       "do_batch_normalization": False, "do_bias": True, "cross_hair": False}
    train_parameters = {"l_r": 0.001, "optimizer": "Adam"}
    hyper_parameters = {"init_parameters": init_parameters, "train_parameters": train_parameters}

    # experiment_1([('xcat', all_xcat_files)], hyper_parameters, k_fold, dimensions, losses, architectures)
    # experiment_1([('ircad', all_ircad_files)], hyper_parameters, k_fold, dimensions, losses, architectures)
    experiment_2([('xcat', all_xcat_files), ('gan', all_gan_files)], [('ircad', all_ircad_files)], hyper_parameters, k_fold, dimensions, losses, architectures)



