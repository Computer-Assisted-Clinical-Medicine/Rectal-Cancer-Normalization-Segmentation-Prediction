import os
import time
import SimpleITK as sitk

import numpy as np
import pandas as pd
import tensorflow as tf

from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.NetworkBasis.util import write_configurations, write_metrics_to_csv, make_csv_file
from SegmentationNetworkBasis.architecture import UNet, VNet, DVN, CombiNet
import SegmentationNetworkBasis.NetworkBasis.image as image
from vesselsegloader import VesselSegRatioLoader
from vesselsegloader import VesselSegLoader
import evaluation

experiment_name = "vessel_segmentation"
if cfg.ONSERVER:
    logs_path = os.path.join("tmp", experiment_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    logs_path = os.path.join("R:\\", experiment_name)


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


def finetuning(model_path, seed=42, **hyper_parameters):
    tf.keras.backend.clear_session()
    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)
    fine_files = pd.read_csv(cfg.fine_csv, dtype=object).as_matrix()
    vald_files = pd.read_csv(cfg.vald_csv, dtype=object).as_matrix()

    training_dataset = VesselSegRatioLoader(name='training_loader') \
        (fine_files, batch_size=cfg.batch_size_train, n_epochs=cfg.training_epochs,
         read_threads=cfg.train_reader_instances)
    validation_dataset = VesselSegRatioLoader(mode=VesselSegRatioLoader.MODES.VALIDATE, name='validation_loader') \
        (vald_files, batch_size=cfg.batch_size_train,
         read_threads=cfg.vald_reader_instances)

    net = hyper_parameters['architecture'](hyper_parameters['loss'], do_finetune=True,
                                           model_path=model_path, **(hyper_parameters["init_parameters"]))

    original_folder = "-".join(os.path.basename(model_path).split('-')[0:2])
    folder_name = original_folder + '-' +_make_folder_name(hyper_parameters, seed)
    write_configurations(hyper_parameters['experiment_path'], folder_name + '-f', net, cfg)
    # Train the network with the dataset iterators
    net.finetune(hyper_parameters['experiment_path'], folder_name, training_dataset, validation_dataset,
              summary_steps_per_epoch=cfg.summary_steps_per_epoch, **(hyper_parameters["train_parameters"]))


def applying(seed=42, model_path='', **hyper_parameters):
    '''!
    do testing

    '''
    tf.keras.backend.clear_session()

    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)
    test_files = pd.read_csv(cfg.test_csv, dtype=object).as_matrix()
    testloader = VesselSegLoader(mode=VesselSegLoader.MODES.APPLY, name='test_loader')

    if hyper_parameters["evaluate_on_finetuned"]:
        cfg.training_epochs = cfg.epochs_for_finetuning
        original_folder = "-".join(os.path.basename(model_path).split('-')[0:2])
        folder_name = original_folder + '-' + _make_folder_name(hyper_parameters, seed) + '-f'
    else:
        cfg.training_epochs = cfg.epochs_for_training
        folder_name = _make_folder_name(hyper_parameters, seed)
    net = hyper_parameters['architecture'](hyper_parameters['loss'], is_training=False, model_path=os.path.join(hyper_parameters['experiment_path'], folder_name), **(hyper_parameters["init_parameters"]))

    for f in test_files:
        test_dataset = testloader(f, batch_size=cfg.batch_size_test, read_threads=cfg.vald_reader_instances)
        net.apply(test_dataset, f)

    tf.keras.backend.clear_session()


def evaluate(seed=42, model_path='', **hyper_parameters):
    '''!
    do testing

    '''

    np.random.seed(42)
    test_files = pd.read_csv(cfg.test_csv, dtype=object).as_matrix()
    if hyper_parameters["evaluate_on_finetuned"]:
        cfg.training_epochs = cfg.epochs_for_finetuning
        original_folder = "-".join(os.path.basename(model_path).split('-')[0:2])
        folder_name = original_folder + '-' + _make_folder_name(hyper_parameters, seed) + '-f'
    else:
        cfg.training_epochs = cfg.epochs_for_training
        folder_name = _make_folder_name(hyper_parameters, seed)

    test_path = os.path.join(hyper_parameters['experiment_path'], folder_name + '_test')
    apply_path = os.path.join(hyper_parameters['experiment_path'], folder_name + '_apply')

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


def fuse_probabilities_over_folds(k_fold, seed=42, **hyper_parameters):
    '''!
    do testing

    '''

    np.random.seed(seed)
    tf.random.set_seed(seed)
    test_files = pd.read_csv(cfg.test_csv, dtype=object).as_matrix()
    cfg.training_epochs = cfg.epochs_for_training
    for file in test_files:
        predictions = []
        for f in range(0, k_fold):
            folder_name = _make_folder_name(hyper_parameters, seed)
            latest_model = tf.train.latest_checkpoint(os.path.join(hyper_parameters['experiment_path'], folder_name, 'model'))
            _, model_base = os.path.split(latest_model)
            name, ext = os.path.splitext(model_base)
            version = name.split('-')[-1]

            apply_path = os.path.join(hyper_parameters['experiment_path'], folder_name + '_apply')

            folder, file_number = os.path.split(file[0])
            prediction_path = os.path.join(apply_path, ('prediction' + '-' + version + '-' + file_number + '.nii'))
            predictions.append(sitk.ReadImage(prediction_path))

        folder_name = _make_folder_name(hyper_parameters, 'fused')
        apply_path = os.path.join(hyper_parameters['experiment_path'], folder_name + '_apply')
        if not os.path.exists(apply_path):
            os.makedirs(apply_path)
        fused_probabilities = 1 / k_fold * sitk.GetArrayFromImage(predictions[0])
        for p in predictions[1:]:
            fused_probabilities = fused_probabilities + 1 / k_fold * sitk.GetArrayFromImage(p)

        data_info = image.get_data_info(p)
        fused_prediction = np.argmax(fused_probabilities, -1)
        pred_img = image.np_array_to_itk_image(fused_prediction, data_info, cfg.label_background_value,
                                               cfg.adapt_resolution, cfg.target_type_label)
        sitk.WriteImage(pred_img, os.path.join(apply_path,
                                ('prediction' + '-' + version + '-' + file_number + '.nii')))
        print('        Fused predictions for ', file_number)


def fuse_probabilities_over_networks(dimensions_and_architectures, seed=42, **hyper_parameters):
    '''!
    do testing

    '''

    np.random.seed(seed)
    tf.random.set_seed(seed)
    test_files = pd.read_csv(cfg.test_csv, dtype=object).as_matrix()
    cfg.training_epochs = cfg.epochs_for_training
    for file in test_files:
        predictions = []
        for d, a in dimensions_and_architectures:
            hyper_parameters["dimensions"] = d
            hyper_parameters['architecture'] = a
            folder_name = _make_folder_name(hyper_parameters, seed)
            latest_model = tf.train.latest_checkpoint(os.path.join(hyper_parameters['experiment_path'], folder_name, 'model'))
            _, model_base = os.path.split(latest_model)
            name, ext = os.path.splitext(model_base)
            version = name.split('-')[-1]

            apply_path = os.path.join(hyper_parameters['experiment_path'], folder_name + '_apply')

            folder, file_number = os.path.split(file[0])
            prediction_path = os.path.join(apply_path, ('prediction' + '-' + version + '-' + file_number + '.nii'))
            predictions.append(sitk.ReadImage(prediction_path))

        hyper_parameters['architecture'] = CombiNet
        hyper_parameters["dimensions"] = 'n'
        folder_name = _make_folder_name(hyper_parameters, 'fused')
        apply_path = os.path.join(hyper_parameters['experiment_path'], folder_name + '_apply')
        if not os.path.exists(apply_path):
            os.makedirs(apply_path)
        fused_probabilities = 1 / len(dimensions_and_architectures) * sitk.GetArrayFromImage(predictions[0])
        for p in predictions[1:]:
            fused_probabilities = fused_probabilities + 1 / len(dimensions_and_architectures) * sitk.GetArrayFromImage(p)

        data_info = image.get_data_info(p)
        fused_prediction = np.argmax(fused_probabilities, -1)
        pred_img = image.np_array_to_itk_image(fused_prediction, data_info, cfg.label_background_value,
                                               cfg.adapt_resolution, cfg.target_type_label)
        sitk.WriteImage(pred_img, os.path.join(apply_path,
                                ('prediction' + '-' + version + '-' + file_number + '.nii')))
        print('        Fused predictions for ', file_number)


def experiment_1(data, hyper_parameters, k_fold, dimensions_and_architectures, losses):
    hyper_parameters["evaluate_on_finetuned"] = False
    # Experiment 1: Train on individual Data sets
    for (data_name, data_set) in data:
        np.random.seed(42)

        hyper_parameters['experiment_path'] = os.path.join(logs_path, 'individual_final5f-fin_' + data_name)
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
                hyper_parameters["dimensions"] = d
                _set_parameters_according_to_dimension(hyper_parameters)
                hyper_parameters['architecture'] = a
                for l in losses:
                    hyper_parameters["loss"] = l

                    # try:
                    #     cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL
                    #     cfg.percent_of_object_samples = 50
                    #     training(**hyper_parameters, seed=f)
                    #     pass
                    # except Exception as err:
                    #     print('Training ' + data_name,
                    #           hyper_parameters['architecture'].get_name() + hyper_parameters['loss'] + 'failed!')
                    #     print(err)
                    #
                    # time.sleep(20)

                    # try:
                    #     cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
                    #     applying(**hyper_parameters, seed=f)
                    # except Exception as err:
                    #     print('Applying ' + data_name,
                    #           hyper_parameters['architecture'].get_name() + hyper_parameters['loss'] + 'failed!')
                    #     print(err)
                    #
                    try:
                        evaluate(**hyper_parameters, seed=f)
                    except Exception as err:
                        print('Evaluating ' + data_name,
                              hyper_parameters['architecture'].get_name() + hyper_parameters['loss'] + 'failed!')
                        print(err)

        # # try:
        evaluation.combine_evaluation_results_from_folds(hyper_parameters['experiment_path'],
                                                         losses, dimensions_and_architectures)
        evaluation.make_boxplot_graphic(hyper_parameters['experiment_path'], dimensions_and_architectures, losses)
        #     pass
        # except Exception as err:
        #     print('Could not combine results!')
        #     print(err)


def experiment_2(data_train, data_fine, hyper_parameters, k_fold, dimensions_and_architectures, losses):
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

            for f in range(0, 1): #
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

                for d, a in dimensions_and_architectures:
                    hyper_parameters["dimensions"] = d
                    hyper_parameters['architecture'] = a
                    _set_parameters_according_to_dimension(hyper_parameters)
                    for l1 in losses:
                        hyper_parameters["loss"] = l1

                        cfg.training_epochs = cfg.epochs_for_training

                        # if f == 0:
                            # try:
                            #     cfg.num_files = len(train_files)
                            #     hyper_parameters['train_parameters']['l_r'] = 0.001
                            #     cfg.training_epochs = 20
                            #     cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL
                            #     cfg.percent_of_object_samples = 50
                            #     training(**hyper_parameters, seed=f)
                            #     pass
                            # except Exception as err:
                            #     print('Training ' + train_data_name,
                            #           hyper_parameters['architecture'].get_name() + hyper_parameters[
                            #               'loss'] + 'failed!')
                            #     print(err)

                        model_path = os.path.join(hyper_parameters['experiment_path'],
                                                                            _make_folder_name(hyper_parameters, f))
                        cfg.training_epochs = cfg.epochs_for_finetuning
                        for l2 in losses:
                            hyper_parameters["loss"] = l2

                            try:
                                cfg.num_files = len(fine_files)
                                hyper_parameters['train_parameters']['l_r'] = 0.0001
                                cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL
                                cfg.percent_of_object_samples = 50
                                finetuning(model_path, **hyper_parameters, seed=f)
                                pass
                            except Exception as err:
                                print('Training ' + train_data_name,
                                      hyper_parameters['architecture'].get_name() + hyper_parameters[
                                          'loss'] + 'failed!')
                                print(err)

                            try:
                                cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
                                applying(seed=f, model_path=model_path, **hyper_parameters)
                            except Exception as err:
                                print('Applying ' + fine_data_name,
                                      hyper_parameters['architecture'].get_name() + hyper_parameters[
                                          'loss'] + 'failed!')
                                print(err)

                            try:
                                evaluate(seed=f, model_path=model_path, **hyper_parameters)
                            except Exception as err:
                                print('Evaluating ' + fine_data_name,
                                      hyper_parameters['architecture'].get_name() + hyper_parameters[
                                          'loss'] + 'failed!')
                                print(err)

            try:
                evaluation.combine_evaluation_results_from_folds(hyper_parameters['experiment_path'],
                    [d for d, a in dimensions_and_architectures], losses, [a for d, a in dimensions_and_architectures],
                                                                 evaluate_on_finetuned=True)
            except Exception as err:
                print('Could not combine results !')
                print(err)

            try:
                evaluation.make_boxplot_graphic(hyper_parameters['experiment_path'], dimensions_and_architectures, losses,
                                                                 evaluate_on_finetuned=True)
            except Exception as err:
                print('Could not generate plots!')
                print(err)


def experiment_3(data_train, data_test, hyper_parameters, k_fold, dimensions_and_architectures, losses):
    # Experiment 3: Apply networks from Experiment 1 as multi-observer
    cfg.training_epochs = cfg.epochs_for_training
    hyper_parameters["evaluate_on_finetuned"] = False
    for (test_data_name, test_data_set) in data_test:
        np.random.seed(42)

        for train_data_name in data_train:
            np.random.seed(42)

            hyper_parameters['experiment_path'] = os.path.join(logs_path,
                                                   'apply_' + train_data_name + '_' + test_data_name)

            test_files = test_data_set
            np.savetxt(cfg.test_csv, test_files, fmt='%s', header='path')

            print('  Data Set ' + train_data_name + '_' + test_data_name + ': '
                  + str(test_files.size) + ' test cases')

            # for d, a in dimensions_and_architectures:
            #     hyper_parameters["dimensions"] = d
            #     hyper_parameters['architecture'] = a
            #     print('-----------------' + a.get_name() + str(d) + '-------------------------')
            #     _set_parameters_according_to_dimension(hyper_parameters)
            #     for l in losses:
            #         hyper_parameters["loss"] = l
            #         cfg.write_probabilities = True
            #         cfg.target_type_label = sitk.sitkVectorFloat32
            #         cfg.adapt_resolution = True
            #         for f in range(0, k_fold):
            #
            #             # try:
            #             #     cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
            #             #     applying(seed=f, **hyper_parameters)
            #             # except Exception as err:
            #             #     print('Applying ' + test_data_name,
            #             #           hyper_parameters['architecture'].get_name() + hyper_parameters[
            #             #               'loss'] + 'failed!')
            #             #     print(err)
            #             pass
            #
            #         cfg.write_probabilities = False
            #         cfg.target_type_label = sitk.sitkUInt8
            #         cfg.adapt_resolution = False
            #         # fuse_probabilities_over_folds(k_fold, seed=f, **hyper_parameters)
            #
            #         try:
            #             evaluate(seed='fused',  **hyper_parameters)
            #         except Exception as err:
            #             print('Evaluating ' + test_data_name,
            #                   hyper_parameters['architecture'].get_name() + hyper_parameters[
            #                       'loss'] + 'failed!')
            #             print(err)
            #
            #         print('------------------------------------------')
            #
            evaluation.combine_evaluation_results_from_folds(hyper_parameters['experiment_path'],
                                                             losses, dimensions_and_architectures)
            evaluation.make_boxplot_graphic(hyper_parameters['experiment_path'], dimensions_and_architectures, losses)


def experiment_4(data, hyper_parameters, k_fold, dimensions_and_architectures, losses):
    hyper_parameters["evaluate_on_finetuned"] = False
    # Experiment 1: Train on individual Data sets
    for (data_name, data_set) in data:
        np.random.seed(42)

        hyper_parameters['experiment_path'] = os.path.join(logs_path, 'apply_' + data_name + '_mullti')
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

            for l in losses:
                hyper_parameters["loss"] = l
                for d, a in dimensions_and_architectures:
                    hyper_parameters["dimensions"] = d
                    _set_parameters_according_to_dimension(hyper_parameters)
                    hyper_parameters['architecture'] = a

                    # cfg.write_probabilities = True
                    # cfg.target_type_label = sitk.sitkVectorFloat32
                    # cfg.adapt_resolution = True
                    #
                    # try:
                    # 	cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
                    # 	applying(seed=f, **hyper_parameters)
                    # except Exception as err:
                    # 	print('Applying ' + data_name,
                    # 			hyper_parameters['architecture'].get_name() + hyper_parameters['loss'] + 'failed!')
                    # 	print(err)
                    # pass

                cfg.write_probabilities = False
                cfg.target_type_label = sitk.sitkUInt8
                cfg.adapt_resolution = False
                # fuse_probabilities_over_networks(dimensions_and_architectures, seed=f, **hyper_parameters)
                hyper_parameters['architecture'] = CombiNet
                hyper_parameters["dimensions"] = 'n'

                try:
                    evaluate(seed='fused',  **hyper_parameters)
                except Exception as err:
                    print('Evaluating ' + data_name,
                          hyper_parameters['architecture'].get_name() + hyper_parameters[
                              'loss'] + 'failed!')
                    print(err)

                print('------------------------------------------')
            
            evaluation.combine_evaluation_results_from_folds(hyper_parameters['experiment_path'],
                                                             losses, (['n', CombiNet],))
            evaluation.make_boxplot_graphic(hyper_parameters['experiment_path'], dimensions_and_architectures, losses)


if __name__ == '__main__':

    np.random.seed(42)

    ircad_csv = 'ircad.csv'
    all_ircad_files = pd.read_csv(ircad_csv, dtype=object).as_matrix()
    btcv_csv = 'btcv.csv'
    all_btcv_files = pd.read_csv(btcv_csv, dtype=object).as_matrix()
    # synth_csv = 'synth.csv'
    # all_synth_files = pd.read_csv(synth_csv, dtype=object).as_matrix()
    # xcat_csv = 'xcat.csv'
    # all_xcat_files = pd.read_csv(xcat_csv, dtype=object).as_matrix()
    # gan_csv = 'gan.csv'
    # all_gan_files = pd.read_csv(gan_csv, dtype=object).as_matrix()

    k_fold = 5
    losses = ['CEL+DICE']
    dimensions_and_architectures = ([2, UNet], [3, VNet])  #[2, UNet],    [3, VNet] ,[2, UNet], [2, VNet], [2, DVN], [3, UNet],[3, VNet], [3, DVN]

    init_parameters = {"regularize": [True, 'L2', 0.0000001], "drop_out": [True, 0.01], "activation": "elu",
                       "do_batch_normalization": False, "do_bias": True, "cross_hair": False}
    train_parameters = {"l_r": 0.001, "optimizer": "Adam"}
    hyper_parameters = {"init_parameters": init_parameters, "train_parameters": train_parameters}

    cfg.epochs_for_training = 80

    # experiment_1([('xcat', all_xcat_files)], hyper_parameters, k_fold, dimensions, losses, architectures)
    # experiment_1([('ircad', all_ircad_files)], hyper_parameters, k_fold, dimensions_and_architectures, losses)

    # cfg.epochs_for_training = 20
    # cfg.epochs_for_finetuning = 20
    # # experiment_2([('gan', all_gan_files), ('xcat', all_xcat_files)], [('ircad', all_ircad_files)], hyper_parameters, k_fold,
    #              dimensions_and_architectures, losses)

    # experiment_3(['ircad'], [('btcv', all_btcv_files)], hyper_parameters, k_fold,
    #              dimensions_and_architectures, losses)

    experiment_4([('ircad', all_ircad_files)], hyper_parameters, k_fold, dimensions_and_architectures, losses)
