import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.NetworkBasis.util import write_configurations
from SegmentationNetworkBasis.architecture import UNet
from SegmentationNetworkBasis.architecture import VNet
from SegmentationNetworkBasis.architecture import FCN
from vesselsegloader import VesselSegRatioLoader

experiment_name = "vessel_segmentation"
if cfg.ONSERVER:
    logs_path = os.path.join("tmp", experiment_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    logs_path = os.path.join("R:", experiment_name)


def training(train_csv, vald_csv, seed=42, **hyper_parameters):
    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)
    train_files = pd.read_csv(train_csv, dtype=object).as_matrix()
    vald_files = pd.read_csv(vald_csv, dtype=object).as_matrix()

    if hyper_parameters['dimensions'] == 2:
        cfg.num_channels = 3
        cfg.train_dim = 256
        cfg.batch_capacity_train = 400
        cfg.batch_capacity_valid = 100
        cfg.train_input_shape = [cfg.train_dim, cfg.train_dim, cfg.num_channels]
        cfg.train_label_shape = [cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
        cfg.batch_size = 4
    elif hyper_parameters['dimensions'] == 3:
        cfg.num_channels = 1
        cfg.train_dim = 128
        cfg.num_slices = 32
        cfg.batch_capacity_train = 100
        cfg.batch_capacity_valid = 50
        cfg.train_input_shape = [cfg.num_slices, cfg.train_dim, cfg.train_dim, cfg.num_channels]
        cfg.train_label_shape = [cfg.num_slices, cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
        print(cfg.train_input_shape, cfg.train_label_shape)
        cfg.batch_size = 2

    training_dataset = VesselSegRatioLoader(name='training_loader') \
        (train_files, batch_size=cfg.batch_size, n_epochs=cfg.training_epochs,
         read_threads=cfg.train_reader_instances)
    validation_dataset = VesselSegRatioLoader(mode=VesselSegRatioLoader.MODES.VALIDATE, name='validation_loader') \
        (vald_files, batch_size=cfg.batch_size,
         read_threads=cfg.vald_reader_instances)

    net = hyper_parameters['architecture'](hyper_parameters['loss'], **(hyper_parameters["init_parameters"]))
    folder_name = net.options['name'] + str(seed) + net.options['loss'] + str(hyper_parameters['dimensions']) + 'D'
    write_configurations(hyper_parameters['experiment_path'], folder_name, net, cfg)
    # Train the network with the dataset iterators
    net.train(hyper_parameters['experiment_path'], folder_name, training_dataset, validation_dataset, summary_steps_per_epoch=cfg.summary_steps_per_epoch)



# def _test(loss, seed, test_files, feed_dict_test):
#     try:
#         # reset all before starting
#         tf.reset_default_graph()
#         g = tf.Graph()
#         with g.as_default():
#             with tf.device('/gpu:0'):
#                 folder_name = 'UNet' + str(seed) + loss + str(cfg.percent_of_object_samples)
#                 newest_model = tf.train.latest_checkpoint(os.path.join(logs_path, folder_name))
#                 print(newest_model)
#                 net = CombiNet(g, loss, is_training=False, model_path=newest_model)
#                 net.test(feed_dict_test, test_files)
#     except RuntimeError as err:
#         print('Testing ' + loss + str(seed) + 'failed!', err)
#
#
# def testing(test_csv, seed=42):
#     '''!
#     do training
#
#     '''
#
#     # inits
#     np.random.seed(seed)
#     tf.set_random_seed(seed)
#     test_files = pd.read_csv(test_csv, dtype=object).as_matrix()
#     feed_dict_test = {}
#
#     _test('GDL', seed, test_files, feed_dict_test)
#     time.sleep(30)


if __name__ == '__main__':

    np.random.seed(42)
    cfg.organ = cfg.ORGANS.LIVER

    ircad_csv = 'ircad.csv'
    all_ircad_files = pd.read_csv(ircad_csv, dtype=object).as_matrix()
    btcv_csv = 'btcv.csv'
    all_btcv_files = pd.read_csv(btcv_csv, dtype=object).as_matrix()
    synth_csv = 'synth.csv'
    all_synth_files = pd.read_csv(synth_csv, dtype=object).as_matrix()
    xcat_csv = 'xcat.csv'
    all_xcat_files = pd.read_csv(xcat_csv, dtype=object).as_matrix()
    gan_csv = 'xcat.csv'
    all_gan_files = pd.read_csv(gan_csv, dtype=object).as_matrix()


    train_csv = 'train.csv'
    vald_csv = 'vald.csv'
    test_csv = 'test.csv'

    init_parameters = {"n_filters": [16, 32, 64, 128, 256], "regularize": [True, 'L2', 0.0000001],
                       "do_batch_normalization": True, "do_bias": False, "cross_hair": False}
    train_parameters = {"l_r": 0.001, "optimizer": "Adam"}
    hyper_parameters = {"init_parameters": init_parameters, "train_parameters": train_parameters}

    # Experiment 1: Train on individual Data sets
    for (data_name, data_set) in [('ircad', all_ircad_files)]:  #, ('btcv', all_btcv_files), ('synth', all_synth_files),
                                  # ('xcat', all_xcat_files), 'gan', all_gan_files]:
        train_files = []
        vald_files = []
        test_files = []
        part_train = int(cfg.data_train_split * data_set.size)
        all_indices = np.random.permutation(range(0, data_set.size))
        train_indices = all_indices[0:part_train]
        vald_indices = all_indices[part_train:part_train+cfg.number_of_vald]
        test_indices = all_indices[part_train+cfg.number_of_vald:]
        train_files.extend(data_set[train_indices])
        vald_files.extend(data_set[vald_indices])
        test_files.extend(data_set[test_indices])
        print('  Data Set: ' + str(train_indices.size) + ' train cases, ' + str(test_indices.size) + ' test cases, ' + str(vald_indices.size) + ' vald cases')

        hyper_parameters['experiment_path'] = os.path.join(logs_path, 'individual_' + data_name)

        np.savetxt(train_csv, train_files, fmt='%s', header='path')
        cfg.num_files = len(train_files)
        np.savetxt(vald_csv, vald_files, fmt='%s', header='path')
        np.savetxt(test_csv, test_files, fmt='%s', header='path')

        cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL
        cfg.percent_of_object_samples = 50
        cfg.training_epochs = 5

        for d in [3]:
            hyper_parameters["dimensions"] = d
            for l in ['GDL']:  # 'CEL', 'TVE', , 'DICE']:
                hyper_parameters["loss"] = l
                for a in [UNet, FCN, VNet]:
                    hyper_parameters['architecture'] = a

                    try:
                        training(train_csv, vald_csv, **hyper_parameters)
                        # testing(test_csv, **hyper_parameters)
                    except Exception as err:
                        print('Training ' + data_name,
                              str(hyper_parameters['architecture']) + hyper_parameters['loss'] + 'failed!')
                        print(err)


    # Experiment 2: Pretrain on synthetic, Finetune on Real