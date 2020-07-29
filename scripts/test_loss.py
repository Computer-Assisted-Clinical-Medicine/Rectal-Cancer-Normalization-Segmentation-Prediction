from SegmentationNetworkBasis.NetworkBasis import loss
from SegLoader import SegLoader
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import SegmentationNetworkBasis.config as cfg


def loss_test(train_csv):
    np.random.seed(42)
    train_files = pd.read_csv(train_csv, dtype=object).as_matrix()

    training_dataset = SegLoader(name='training_loader') \
        (train_files, batch_size=cfg.batch_size_train, n_epochs=1,
         read_threads=cfg.train_reader_instances)

    train_writer = tf.summary.create_file_writer(os.path.join('..\\tmp', 'loss_test'))
    with train_writer.as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)

        for x_train, y_train in training_dataset:
            y_1 = tf.greater_equal(tf.nn.dilation2d(input=tf.expand_dims(y_train[:, :, :, 1], -1),
                                                    filters=tf.expand_dims([[0, 0, 1.0, 0, 0], [0, 1.0, 1.0, 1.0, 0],
                                                                            [1.0, 1.0, 1.0, 1.0, 1.0],
                                                                            [0, 1.0, 1.0, 1.0, 0], [0, 0, 1.0, 0, 0]],
                                                                           -1),
                                                    strides=[1, 1, 1, 1], padding='SAME', data_format="NHWC",
                                                    dilations=[1, 2, 2, 1]), 2)
            y_2 = tf.greater_equal(tf.nn.dilation2d(input=tf.expand_dims(y_train[:, :, :, 2], -1),
                                                          filters=tf.expand_dims(
                                                              [[0, 0, 1.0, 0, 0], [0, 1.0, 1.0, 1.0, 0],
                                                               [1.0, 1.0, 1.0, 1.0, 1.0], [0, 1.0, 1.0, 1.0, 0],
                                                               [0, 0, 1.0, 0, 0]], -1),
                                                          strides=[1, 1, 1, 1], padding='SAME', data_format="NHWC",
                                                          dilations=[1, 2, 2, 1]), 2)
            background = tf.logical_not(tf.logical_and(y_1, y_2))
            y_other = tf.cast(tf.concat([background, y_1, y_2], axis=-1, name='stack'), tf.float32)

            y_1 = tf.expand_dims(y_train[:, :, :, 1], -1) * 0.0
            y_2 = y_1
            background = y_1 + 1.0
            y_wrong = tf.cast(tf.concat([background, y_1, y_2], axis=-1, name='stack'), tf.float32)

            y_1 = tf.greater(tf.expand_dims(y_train[:, :, :, 2], -1), 0)
            y_2 = tf.greater(tf.expand_dims(y_train[:, :, :, 1], -1), 0)
            background = tf.logical_not(tf.logical_and(y_1, y_2))
            y_inverse = tf.cast(tf.concat([background, y_1, y_2], axis=-1, name='stack'), tf.float32)

            tf.summary.image('train_img',
                                 tf.cast(tf.gather(x_train, [0, cfg.batch_size_train - 1]) * 255 / 2,
                                         tf.uint8), global_step, 2)

            tf.summary.image('train_seg_lbl',
                             tf.expand_dims(tf.cast(tf.argmax(tf.gather(y_train, [0, cfg.batch_size_train - 1]), -1) * 255 / 2, tf.uint8),
                                            axis=-1), global_step, 2)
            tf.summary.histogram('train_seg_lbl', y_train, step=global_step)
            tf.summary.image('train_seg_other',
                             tf.expand_dims(tf.cast(tf.argmax(tf.gather(y_other, [0, cfg.batch_size_train - 1]), -1) * 255 / 2, tf.uint8),
                                            axis=-1), global_step, 2)
            tf.summary.histogram('train_seg_other', y_other, step=global_step)
            tf.summary.image('train_seg_wrong',
                             tf.expand_dims(tf.cast(tf.argmax(tf.gather(y_wrong, [0, cfg.batch_size_train - 1]), -1) * 255 / 2, tf.uint8),
                                 axis=-1), global_step, 2)
            tf.summary.histogram('train_seg_wrong', y_wrong, step=global_step)
            tf.summary.image('train_seg_inverse',
                             tf.expand_dims(
                                 tf.cast(tf.argmax(tf.gather(y_inverse, [0, cfg.batch_size_train - 1]), -1) * 255 / 2,
                                         tf.uint8),
                                 axis=-1), global_step, 2)
            tf.summary.histogram('train_seg_inverse', y_inverse, step=global_step)



            with tf.name_scope('01_perfect_losses'):
                tf.summary.scalar('perfect_WCEL', loss.weighted_categorical_crossentropy_with_fpr_loss(y_train, y_train),
                                  step=global_step)
                tf.summary.scalar('perfect_generalized_dice', loss.generalized_dice_loss(y_train, y_train), step=global_step)
                tf.summary.scalar('perfect_dice', loss.dice_loss(y_train, y_train), step=global_step)
                tf.summary.scalar('perfect_tversky', loss.tversky_loss(y_train, y_train), step=global_step)
                tf.summary.scalar('perfect_weighted dice', loss.weighted_dice_loss(y_train, y_train), step=global_step)
                tf.summary.scalar('perfect_categorical_cross_entropy_loss',
                                  loss.categorical_cross_entropy_loss(y_train, y_train), step=global_step)
            with tf.name_scope('02_wrong_losses'):
                tf.summary.scalar('wrong_WCEL', loss.weighted_categorical_crossentropy_with_fpr_loss(y_train, y_wrong),
                                  step=global_step)
                tf.summary.scalar('wrong_generalized_dice', loss.generalized_dice_loss(y_train, y_wrong), step=global_step)
                tf.summary.scalar('wrong_dice', loss.dice_loss(y_train, y_wrong), step=global_step)
                tf.summary.scalar('wrong_tversky', loss.tversky_loss(y_train, y_wrong), step=global_step)
                tf.summary.scalar('wrong_weighted dice', loss.weighted_dice_loss(y_train, y_wrong), step=global_step)
                tf.summary.scalar('wrong_categorical_cross_entropy_loss',
                                  loss.categorical_cross_entropy_loss(y_train, y_wrong), step=global_step)
            with tf.name_scope('03_other_losses'):
                tf.summary.scalar('other_WCEL', loss.weighted_categorical_crossentropy_with_fpr_loss(y_train, y_other),
                                  step=global_step)
                tf.summary.scalar('other_generalized_dice', loss.generalized_dice_loss(y_train, y_other), step=global_step)
                tf.summary.scalar('other_dice', loss.dice_loss(y_train, y_other), step=global_step)
                tf.summary.scalar('other_tversky', loss.tversky_loss(y_train, y_other), step=global_step)
                tf.summary.scalar('other_weighted dice', loss.weighted_dice_loss(y_train, y_other), step=global_step)
                tf.summary.scalar('other_categorical_cross_entropy_loss',
                                  loss.categorical_cross_entropy_loss(y_train, y_other), step=global_step)
            with tf.name_scope('04_inverse_losses'):
                tf.summary.scalar('inverse_WCEL', loss.weighted_categorical_crossentropy_with_fpr_loss(y_train, y_inverse),
                                  step=global_step)
                tf.summary.scalar('inverse_generalized_dice', loss.generalized_dice_loss(y_train, y_inverse), step=global_step)
                tf.summary.scalar('inverse_dice', loss.dice_loss(y_train, y_inverse), step=global_step)
                tf.summary.scalar('inverse_tversky', loss.tversky_loss(y_train, y_inverse), step=global_step)
                tf.summary.scalar('inverse_weighted dice', loss.weighted_dice_loss(y_train, y_inverse), step=global_step)
                tf.summary.scalar('inverse_categorical_cross_entropy_loss',
                                  loss.categorical_cross_entropy_loss(y_train, y_inverse), step=global_step)

            global_step = global_step + 1

        print('-----------------------------------------------------------')


if __name__ == '__main__':
    cfg.batch_capacity = cfg.batch_capacity_train // 4
    train_csv = '../ircad.csv'

    cfg.num_channels = 3
    cfg.train_dim = 256
    cfg.samples_per_volume = 150
    cfg.batch_capacity_train = 300
    cfg.batch_capacity_valid = 150
    cfg.train_input_shape = [cfg.train_dim, cfg.train_dim, cfg.num_channels]
    cfg.train_label_shape = [cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
    print('   Train Shapes: ', cfg.train_input_shape, cfg.train_label_shape)
    cfg.test_dim = 512
    cfg.test_data_shape = [cfg.test_dim, cfg.test_dim, cfg.num_channels]
    cfg.test_label_shape = [cfg.test_dim, cfg.test_dim, cfg.num_classes_seg]
    print('   Test Shapes: ', cfg.test_data_shape, cfg.test_label_shape)
    cfg.batch_size_train = 16
    cfg.batch_size_test = 1

    loss_test(train_csv)