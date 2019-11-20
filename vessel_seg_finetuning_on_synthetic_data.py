import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.NetworkBasis.util import write_configurations, write_metrics_to_csv, make_csv_file
from SegmentationNetworkBasis.architecture import UNet, VNet, FCN
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
    epochs = cfg.training_epochs // 100

    folder_name = "-".join([hyper_parameters['architecture'].get_name() + str(hyper_parameters['dimensions']) + 'D',
                            hyper_parameters['loss'], str(epochs), str(seed)])

    return folder_name


def _set_parameters_according_to_dimension(hyper_parameters):
    if hyper_parameters['dimensions'] == 2:
        cfg.num_channels = 3
        cfg.train_dim = 256
        cfg.samples_per_volume = 150
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
        cfg.samples_per_volume = 50
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


def training(train_csv, vald_csv, seed=42, **hyper_parameters):
    tf.keras.backend.clear_session()
    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)
    train_files = pd.read_csv(train_csv, dtype=object).as_matrix()
    vald_files = pd.read_csv(vald_csv, dtype=object).as_matrix()

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


def testing(test_csv, seed=42, **hyper_parameters):
    '''!
    do testing

    '''
    tf.keras.backend.clear_session()

    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)
    test_files = pd.read_csv(test_csv, dtype=object).as_matrix()
    testloader = VesselSegLoader(mode=VesselSegLoader.MODES.APPLY, name='test_loader')

    folder_name = _make_folder_name(hyper_parameters, seed)
    net = hyper_parameters['architecture'](hyper_parameters['loss'], is_training=False, model_path=os.path.join(hyper_parameters['experiment_path'], folder_name), **(hyper_parameters["init_parameters"]))

    for f in test_files:
        test_dataset = testloader(f, batch_size=cfg.batch_size_test, read_threads=cfg.vald_reader_instances)
        net.apply(test_dataset, f)


def evaluate(test_csv, seed=42, **hyper_parameters):
    '''!
    do testing

    '''

    np.random.seed(seed)
    test_files = pd.read_csv(test_csv, dtype=object).as_matrix()
    folder_name = _make_folder_name(hyper_parameters, seed)
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

if __name__ == '__main__':

    np.random.seed(42)
    cfg.organ = cfg.ORGANS.LIVER

    ircad_csv = 'ircad.csv'
    all_ircad_files = pd.read_csv(ircad_csv, dtype=object).as_matrix()
    # btcv_csv = 'btcv.csv'
    # all_btcv_files = pd.read_csv(btcv_csv, dtype=object).as_matrix()
    # synth_csv = 'synth.csv'
    # all_synth_files = pd.read_csv(synth_csv, dtype=object).as_matrix()
    xcat_csv = 'xcat.csv'
    all_xcat_files = pd.read_csv(xcat_csv, dtype=object).as_matrix()
    # gan_csv = 'xcat.csv'
    # all_gan_files = pd.read_csv(gan_csv, dtype=object).as_matrix()


    train_csv = 'train.csv'
    vald_csv = 'vald.csv'
    test_csv = 'test.csv'
    k_fold = 4

    init_parameters = {"regularize": [True, 'L2', 0.0000001], "drop_out": [True, 0.2],
                       "do_batch_normalization": True, "do_bias": False, "cross_hair": False}
    train_parameters = {"l_r": 0.001, "optimizer": "Adam"}
    hyper_parameters = {"init_parameters": init_parameters, "train_parameters": train_parameters}

    # Experiment 1: Train on individual Data sets
    for (data_name, data_set) in [('xcat', all_xcat_files), ('ircad', all_ircad_files)]:  #, ('btcv', all_btcv_files), ('synth', all_synth_files),
                                  # , 'gan', all_gan_files]:

        hyper_parameters['experiment_path'] = os.path.join(logs_path, 'individual_2D3D_' + data_name)
        train_files = []
        vald_files = []
        test_files = []
        part_train = int(cfg.data_train_split * data_set.size)
        all_indices = np.random.permutation(range(0, data_set.size))
        test_folds = np.array_split(all_indices, k_fold)

        for f in range(k_fold):
            test_indices = test_folds[f]
            remaining_indices = np.setdiff1d(all_indices, test_folds[f])
            vald_indices = remaining_indices[:cfg.number_of_vald]
            train_indices = remaining_indices[cfg.number_of_vald:]

            train_files = data_set[train_indices]
            vald_files = data_set[vald_indices]
            test_files = data_set[test_indices]

            # train_indices = all_indices[0:part_train]
            # vald_indices = all_indices[part_train:part_train + cfg.number_of_vald]
            # test_indices = all_indices[part_train + cfg.number_of_vald:]
            # train_files.extend(data_set[train_indices])
            # vald_files.extend(data_set[vald_indices])
            # test_files.extend(data_set[test_indices])
            #
            # train_files.extend(data_set[train_indices])
            # vald_files.extend(data_set[vald_indices])
            # test_files.extend(data_set[test_indices])

            np.savetxt(train_csv, train_files, fmt='%s', header='path')
            np.savetxt(vald_csv, vald_files, fmt='%s', header='path')
            np.savetxt(test_csv, test_files, fmt='%s', header='path')

            cfg.num_files = len(train_files)

            print('  Data Set ' + data_name + str(f) + ': ' + str(train_indices.size) + ' train cases, ' + str(test_indices.size)
                  + ' test cases, ' + str(vald_indices.size) + ' vald cases')

            cfg.training_epochs = 60

            for d in [2, 3]:
                hyper_parameters["dimensions"] = d
                _set_parameters_according_to_dimension(hyper_parameters)
                for l in ['WDL']:  # 'CEL', 'TVE', , 'DICE']:
                    hyper_parameters["loss"] = l
                    for a in [FCN, UNet, VNet]:
                        hyper_parameters['architecture'] = a

                        try:
                            cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL
                            cfg.percent_of_object_samples = 50
                            training(train_csv, vald_csv, **hyper_parameters, seed=f)
                            pass
                        except Exception as err:
                            print('Training ' + data_name,
                                  hyper_parameters['architecture'].get_name() + hyper_parameters['loss'] + 'failed!')
                            print(err)

                        try:
                            cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
                            testing(test_csv, **hyper_parameters, seed=f)
                        except Exception as err:
                            print('Testing ' + data_name,
                                  hyper_parameters['architecture'].get_name() + hyper_parameters['loss'] + 'failed!')
                            print(err)

                        try:
                            evaluate(test_csv, **hyper_parameters, seed=f)
                        except Exception as err:
                            print('Testing ' + data_name,
                                  hyper_parameters['architecture'].get_name() + hyper_parameters['loss'] + 'failed!')
                            print(err)




    # Experiment 2: Pretrain on synthetic, Finetune on Real