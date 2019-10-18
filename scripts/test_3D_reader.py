from LiverSeg.liversegloader import LiverSegLoader
from LiverSeg.liversegloader import LiverSegRatioLoader
import SimpleITK as sitk
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from LiverSeg.SegmentationNetworkBasis import config as cfg

if cfg.ONSERVER:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto(intra_op_parallelism_threads=6,
                            inter_op_parallelism_threads=6, allow_soft_placement=True,
                            device_count={'CPU': 1, 'GPU': 1})
else:
    pass


def _reader_test(training_dataset, mode):
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)

    valid_writer = tf.summary.create_file_writer(os.path.join('..\\tmp', 'reader_test3D', ''))
    with valid_writer.as_default():
            for x_train, y_train in training_dataset:
                try:
                    for b in range(cfg.batch_size):
                        sample_img = tf.gather(x_train, [b])
                        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(sample_img.numpy())),
                                        'train_vol' + '-' + str(b) + '.nii')
                        sample_img = tf.gather(y_train[:, :, :, :, 1], [b])
                        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(sample_img.numpy())),
                                        'label_vol' + '-' + str(b) + '.nii')

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
    cfg.batch_size = 2

    cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL
    cfg.normalizing_method = cfg.NORMALIZING.WINDOW

    run_liver_ratio_test(train_csv, LiverSegLoader.MODES.TRAIN)
