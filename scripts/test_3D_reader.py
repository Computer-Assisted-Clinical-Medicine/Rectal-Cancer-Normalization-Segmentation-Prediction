from SegLoader import SegLoader
from SegLoader import SegRatioLoader
import tensorflow as tf
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os
from SegmentationNetworkBasis import config as cfg

experiment_name = 'reader_test3D'
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
            for x_train, y_train in training_dataset:
                try:
                    with tf.name_scope('01_Input_and_Predictions'):
                        tf.summary.image('train_seg_lbl', tf.expand_dims(
                            tf.cast(tf.argmax(tf.gather(tf.squeeze(y_train[:, y_train.shape[1] // 2, :, :, :]),
                                                        [0, cfg.batch_size_train - 1]), -1) * (
                                             255 // (cfg.num_classes_seg - 1)), tf.uint8), axis=-1),
                                         global_step, 2)

                        tf.summary.image('train_img', tf.cast((tf.gather(x_train[:, x_train.shape[1] // 2, :, :],
                                                                         [0, cfg.batch_size_train - 1]) + 1) * 255 / 2,
                                                              tf.uint8), global_step, 2)

                    for b in range(cfg.batch_size_train):
                        sample_img = tf.gather(x_train, [b])
                        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(sample_img.numpy())),
                                        os.path.join(logs_path, 'train_vol' + '-' + str(b) + '.nii'))
                        sample_img = tf.cast(tf.argmax(tf.gather(y_train, [b]), -1), tf.uint8)
                        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(sample_img.numpy())),
                                        os.path.join(logs_path, 'label_vol' + '-' + str(b) + '.nii'))

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


def run_vessel_ratio_test(train_csv, mode):
    np.random.seed(42)
    train_files = pd.read_csv(train_csv, dtype=object).as_matrix()
    if mode == SegLoader.MODES.APPLY:
        train_files = train_files[0]
    else:
        train_files = train_files[0:2]
    loader_name = 'lits_liver_ratio_loader'

    training_dataset = SegRatioLoader(name=loader_name, mode=mode)\
            (train_files, batch_size=cfg.batch_size_train, n_epochs=cfg.training_epochs, read_threads=cfg.vald_reader_instances)

    print('Testing: ' + loader_name + ' ' + str(cfg.random_sampling_mode))
    _reader_test(training_dataset, mode)


if __name__ == '__main__':
    train_csv = '../btcv.csv'
    print('Loading training file names from %s' % train_csv)
    cfg.training_epochs = 3
    cfg.num_channels = 1
    cfg.train_dim = 96
    cfg.num_slices = 32
    cfg.batch_capacity_train = 40
    cfg.batch_capacity_valid = 50
    cfg.train_input_shape = [cfg.num_slices, cfg.train_dim, cfg.train_dim, cfg.num_channels]
    cfg.train_label_shape = [cfg.num_slices, cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
    print(cfg.train_input_shape, cfg.train_label_shape)
    cfg.batch_size_train = 2

    cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL
    cfg.normalizing_method = cfg.NORMALIZING.WINDOW

    run_vessel_ratio_test(train_csv, SegLoader.MODES.TRAIN)
