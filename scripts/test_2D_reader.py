from vesselsegloader import VesselSegLoader
from vesselsegloader import VesselSegRatioLoader
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from SegmentationNetworkBasis import config as cfg

experiment_name = 'reader_test2D'
logs_path = os.path.join('..\\tmp', experiment_name)

if cfg.ONSERVER:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto(intra_op_parallelism_threads=6,
                            inter_op_parallelism_threads=6, allow_soft_placement=True,
                            device_count={'CPU': 1, 'GPU': 1})
else:
    pass


def _reader_test(training_dataset, mode):
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)

    valid_writer = tf.summary.create_file_writer(logs_path)
    with valid_writer.as_default():
        with tf.name_scope('01_Input_and_Predictions'):
            if mode == VesselSegLoader.MODES.APPLY:
                for x_train in training_dataset:
                    try:
                        print('Finished Batch: ', x_train.shape)

                        if cfg.normalizing_method == cfg.NORMALIZING.WINDOW:
                            tf.summary.image('train_img_WINDOW',
                                             tf.cast((tf.gather(x_train, [0, cfg.batch_size - 1]) + 1) * 255 / 2, tf.uint8),
                                             global_step, 1)
                        else:
                            tf.summary.image('train_img_MEAN_STD',
                                             tf.cast((tf.gather(x_train, [0, cfg.batch_size - 1]) + 1) * 255 / 4.5,
                                                     tf.uint8),
                                             global_step, 1)

                        if cfg.normalizing_method == cfg.NORMALIZING.WINDOW:
                            tf.summary.histogram('train_data_WINDOW', x_train, global_step, 20)
                        else:
                            tf.summary.histogram('train_data_MEAN_STD', x_train, global_step, 20)

                        global_step = global_step + 1
                        if global_step >= 200:
                            break
                    except tf.errors.OutOfRangeError:
                        break

            else:
                for x_train, y_train in training_dataset:
                    try:
                        if cfg.normalizing_method == cfg.NORMALIZING.WINDOW:
                            tf.summary.image('train_img_WINDOW',
                                             tf.cast((tf.gather(x_train, [0, cfg.batch_size - 1]) + 1) * 255 / 2, tf.uint8),
                                             global_step, 2)
                        else:
                            tf.summary.image('train_img_MEAN_STD',
                                             tf.cast((tf.gather(x_train, [0, cfg.batch_size - 1]) + 1) * 255 / 4.5, tf.uint8),
                                             global_step, 2)

                        tf.summary.image('train_art_lbl',
                                         tf.expand_dims(
                                             tf.cast(tf.gather(y_train, [0, cfg.batch_size - 1])[:, :, :, 1] * 255, tf.uint8),
                                             axis=-1), global_step, 2)
                        tf.summary.image('train_vein_lbl',
                                         tf.expand_dims(
                                             tf.cast(tf.gather(y_train, [0, cfg.batch_size - 1])[:, :, :, 2] * 255,
                                                     tf.uint8),
                                             axis=-1), global_step, 2)
                        tf.summary.image('train_seg_lbl',
                                         tf.expand_dims(
                                             tf.cast(tf.argmax(tf.gather(y_train, [0, cfg.batch_size - 1]),-1) *(255//(cfg.num_classes_seg-1)),
                                                     tf.uint8), axis=-1), global_step, 2)

                        if cfg.normalizing_method == cfg.NORMALIZING.WINDOW:
                            tf.summary.histogram('train_data_WINDOW', x_train, global_step, 20)
                        else:
                            tf.summary.histogram('train_data_MEAN_STD', x_train, global_step, 20)

                        tf.summary.histogram('train_label', y_train, global_step, 20)
                        print('Finished Batch: ', x_train.shape, y_train.shape)
                        global_step = global_step + 1
                    except tf.errors.OutOfRangeError:
                        break

    print('-----------------------------------------------------------')


def run_vessel_test(train_csv, mode):
    np.random.seed(42)
    train_files = pd.read_csv(train_csv, dtype=object).as_matrix()
    if mode == VesselSegLoader.MODES.APPLY:
        train_files = train_files[0]
    else:
        train_files = train_files[0:2]
    loader_name = 'vessel_loader'

    training_dataset = VesselSegLoader(name=loader_name, mode=mode)\
            (train_files, batch_size=cfg.batch_size, n_epochs=cfg.training_epochs, read_threads=cfg.vald_reader_instances)

    print('Testing: ' + loader_name + ' ' + str(cfg.random_sampling_mode))
    _reader_test(training_dataset, mode)


def run_vessel_ratio_test(train_csv, mode):
    np.random.seed(42)
    train_files = pd.read_csv(train_csv, dtype=object).as_matrix()
    if mode == VesselSegLoader.MODES.APPLY:
        train_files = train_files[0]
    else:
        train_files = train_files[0:2]
    loader_name = 'vessel_ratio_loader'

    training_dataset = VesselSegRatioLoader(name=loader_name, mode=mode)\
            (train_files, batch_size=cfg.batch_size, n_epochs=cfg.training_epochs, read_threads=cfg.vald_reader_instances)

    print('Testing: ' + loader_name + ' ' + str(cfg.random_sampling_mode))
    _reader_test(training_dataset, mode)


if __name__ == '__main__':
    train_csv = '../ircad.csv'
    print('Loading training file names from %s' % train_csv)
    cfg.training_epochs = 3

    cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL
    cfg.normalizing_method = cfg.NORMALIZING.WINDOW

    run_vessel_ratio_test(train_csv, VesselSegLoader.MODES.TRAIN)
    run_vessel_ratio_test(train_csv, VesselSegLoader.MODES.VALIDATE)

    run_vessel_test(train_csv, VesselSegLoader.MODES.VALIDATE)
    run_vessel_test(train_csv, VesselSegLoader.MODES.TRAIN)

    cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
    run_vessel_test(train_csv, VesselSegLoader.MODES.APPLY)

    cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_MUSTD
    cfg.normalizing_method = cfg.NORMALIZING.MEAN_STD

    run_vessel_ratio_test(train_csv, VesselSegLoader.MODES.VALIDATE)
    run_vessel_ratio_test(train_csv, VesselSegLoader.MODES.TRAIN)

    run_vessel_test(train_csv, VesselSegLoader.MODES.VALIDATE)
    run_vessel_test(train_csv, VesselSegLoader.MODES.TRAIN)

    cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
    run_vessel_test(train_csv, VesselSegLoader.MODES.APPLY)
