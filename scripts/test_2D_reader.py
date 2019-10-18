from LiverSeg.liversegloader import LiverSegLoader
from LiverSeg.liversegloader import LiverSegRatioLoader
from LiverSeg.liversegloader import TumorSegLoader
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from SegmentationNetworkBasis import config as cfg

if cfg.ONSERVER:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto(intra_op_parallelism_threads=6,
                            inter_op_parallelism_threads=6, allow_soft_placement=True,
                            device_count={'CPU': 1, 'GPU': 1})
else:
    pass


def _reader_test(training_dataset, mode):
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)

    valid_writer = tf.summary.create_file_writer(os.path.join('..\\tmp', 'reader_test', ''))
    with valid_writer.as_default():
        if mode == LiverSegLoader.MODES.APPLY:
            for x_train in training_dataset:
                try:
                    print('Finished Batch: ', x_train.shape)

                    if cfg.normalizing_method == cfg.NORMALIZING.WINDOW:
                        tf.summary.image('train_img_WINDOW',
                                         tf.cast((tf.gather(x_train, [0, cfg.batch_size - 1]) + 1) * 255 / 2, tf.uint8),
                                         global_step, 1)
                        # tf.summary.image('train_img_WINDOW',
                        #                  tf.cast((x_train + 1) * 255 / 2, tf.uint8), global_step, 1)
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

                    tf.summary.image('train_seg_lbl',
                                     tf.expand_dims(
                                         tf.cast(tf.gather(y_train, [0, cfg.batch_size - 1])[:, :, :, 1] * 255, tf.uint8),
                                         axis=-1), global_step, 2)

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


def run_liver_test(train_csv, mode):
    np.random.seed(42)
    train_files = pd.read_csv(train_csv, dtype=object).as_matrix()
    if mode == LiverSegLoader.MODES.APPLY:
        train_files = train_files[0]
    else:
        train_files = train_files[0:2]
    loader_name = 'lits_liver_loader'

    training_dataset = LiverSegLoader(name=loader_name, mode=mode)\
            (train_files, batch_size=cfg.batch_size, n_epochs=cfg.training_epochs, read_threads=cfg.vald_reader_instances)

    print('Testing: ' + loader_name + ' ' + str(cfg.random_sampling_mode))
    _reader_test(training_dataset, mode)


def run_liver_ratio_test(train_csv, mode):
    np.random.seed(42)
    train_files = pd.read_csv(train_csv, dtype=object).as_matrix()
    if mode == LiverSegLoader.MODES.APPLY:
        train_files = train_files[0]
    else:
        train_files = train_files[0:2]
    loader_name = 'lits_liver_ratio_loader'

    training_dataset = LiverSegRatioLoader(name=loader_name, mode=mode)\
            (train_files, batch_size=cfg.batch_size, n_epochs=cfg.training_epochs, read_threads=cfg.vald_reader_instances)

    print('Testing: ' + loader_name + ' ' + str(cfg.random_sampling_mode))
    _reader_test(training_dataset, mode)


def run_tumor_test(train_csv):
    np.random.seed(42)
    train_files = pd.read_csv(train_csv, dtype=object).as_matrix()
    loader_name = 'lits_tumor_loader'

    training_dataset = TumorSegLoader(name=loader_name)\
            (train_files, batch_size=cfg.batch_size, n_epochs=cfg.training_epochs, read_threads=cfg.vald_reader_instances)

    print('Testing: ' + loader_name + ' ' + str(cfg.random_sampling_mode))
    _reader_test(training_dataset)


if __name__ == '__main__':
    train_csv = '../btcv.csv'
    print('Loading training file names from %s' % train_csv)
    cfg.training_epochs = 3

    cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL
    cfg.normalizing_method = cfg.NORMALIZING.WINDOW

    # run_liver_ratio_test(train_csv, LiverSegLoader.MODES.TRAIN)
    # run_liver_ratio_test(train_csv, LiverSegLoader.MODES.VALIDATE)
    #
    # run_liver_test(train_csv, LiverSegLoader.MODES.VALIDATE)
    # run_liver_test(train_csv, LiverSegLoader.MODES.TRAIN)
    #
    # cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
    # run_liver_test(train_csv, LiverSegLoader.MODES.APPLY)

    cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_MUSTD
    cfg.normalizing_method = cfg.NORMALIZING.MEAN_STD

    # run_liver_ratio_test(train_csv, LiverSegLoader.MODES.VALIDATE)
    # run_liver_ratio_test(train_csv, LiverSegLoader.MODES.TRAIN)

    run_liver_test(train_csv, LiverSegLoader.MODES.VALIDATE)
    run_liver_test(train_csv, LiverSegLoader.MODES.TRAIN)

    cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
    run_liver_test(train_csv, LiverSegLoader.MODES.APPLY)
