import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import tensorflow as tf

import create_test_files
import seg_data_loader
from SegmentationNetworkBasis import config as cfg

# TODO: add tests for apply loader

def set_parameters_according_to_dimension(dimension, num_channels, preprocessed_dir):
    if dimension == 2:
        cfg.num_channels = num_channels
        #cfg.train_dim = 256
        cfg.samples_per_volume = 160
        cfg.batch_capacity_train = 750
        cfg.train_input_shape = [cfg.train_dim, cfg.train_dim, cfg.num_channels]
        cfg.train_label_shape = [cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
        #cfg.test_dim = 512
        cfg.test_data_shape = [cfg.test_dim, cfg.test_dim, cfg.num_channels]
        cfg.test_label_shape = [cfg.test_dim, cfg.test_dim, cfg.num_classes_seg]
        cfg.batch_size_train = 64
    elif dimension == 3:
        cfg.num_channels = num_channels
        #cfg.train_dim = 128
        cfg.samples_per_volume = 80
        cfg.batch_capacity_train = 250
        cfg.num_slices_train = 8
        cfg.train_input_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_channels]
        cfg.train_label_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
        #cfg.test_dim = 512
        cfg.num_slices_test = 8
        cfg.test_data_shape = [cfg.num_slices_test, cfg.test_dim, cfg.test_dim, cfg.num_channels]
        cfg.test_label_shape = [cfg.num_slices_test, cfg.test_dim, cfg.test_dim, cfg.num_classes_seg]
        cfg.batch_size_train = 16 #Otherwise, VNet 3D fails

    # set config
    if not preprocessed_dir.exists():
        preprocessed_dir.mkdir(parents=True)
    cfg.preprocessed_dir = str(preprocessed_dir)
    cfg.normalizing_method == cfg.NORMALIZING.PERCENT5

    return

@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('name', ['train', 'vald'])
@pytest.mark.parametrize('module', [seg_data_loader])
def test_functions(dimension, name, module):

    test_dir = Path('test_data')

    set_parameters_according_to_dimension(dimension, 2, test_dir/'data_preprocessed')

    set_seeds()

    #generate loader
    data_loader = get_loader(name, module)

    # get names from csv
    file_list, files_list_b = load_dataset(test_dir)

    print(f'Loading Dataset {name}.')

    print('\tLoad a numpy sample')
    data_read = data_loader._read_file_and_return_numpy_samples(files_list_b[0])
    if name == 'test':
        samples = data_read[0]
    else:
        samples, labels = data_read
        print(f'\tSamples from foreground shape: {samples.shape}')
        print(f'\tLabels from foreground shape: {labels.shape}')

        assert(samples.shape[:-1] == labels.shape[:-1])

        nan_slices = np.all(np.isnan(samples), axis=(1,2,3))
        assert not np.any(nan_slices), f'{nan_slices.sum()} sample slices contain NANs'

        nan_slices = np.all(np.isnan(labels), axis=(1,2,3))
        assert not np.any(nan_slices), f'{nan_slices.sum()} label slices contain NANs'

    # call the wrapper function
    data_loader._read_wrapper(id_data_set=tf.squeeze(tf.convert_to_tensor(file_list[0], dtype=tf.string)))


@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('name', ['train', 'vald'])
@pytest.mark.parametrize('module', [seg_data_loader])
def test_wrapper(dimension, name, module):
    n_epochs = 1

    test_dir = Path('test_data')

    set_parameters_according_to_dimension(dimension, 2, test_dir/'data_preprocessed')

    set_seeds()

    #generate loader
    data_loader = get_loader(name, module)

    # get names from csv
    file_list, _ = load_dataset(test_dir)

    data_file, _ = data_loader._get_filenames(str(file_list[0]))
    first_image = sitk.GetArrayFromImage(sitk.ReadImage(data_file))

    print(f'Loading Dataset {name}.')

    # call the loader
    if name == 'train':
        dataset = data_loader(
            file_list,
            batch_size=cfg.batch_size_train,
            n_epochs=n_epochs,
            read_threads=cfg.train_reader_instances
        )
    elif name == 'test':
        dataset = data_loader(
            file_list[0], # only pass one file to the test loader
        )
    else:
        dataset = data_loader(
            file_list,
            batch_size=cfg.batch_size_train,
            read_threads=cfg.vald_reader_instances
        )

    print('\tLoad samples using the data loader')

    # count iterations
    counter = 0
    # save fraction of slices with samples
    n_objects = []
    n_background = []

    for sample in dataset:
        # test set only contains samples, not labels
        if name == 'test':
            x_t = sample
        else:
            x_t, y_t = sample

        # check shape
        if name != 'test':
            assert(cfg.batch_size_train == x_t.shape[0])
            assert(cfg.num_channels == x_t.shape[-1])
            assert(cfg.batch_size_train == y_t.shape[0])
        else:
            if dimension == 2:
                assert(first_image.shape[0] == x_t.numpy().shape[0])

        # look for nans
        nan_slices = np.all(np.isnan(x_t.numpy()), axis=(1,2,3))
        if np.any(nan_slices):
            print(f'{nan_slices.sum()} sample slices only contain NANs')

        if name != 'test':
            nan_slices = np.all(np.isnan(y_t.numpy()), axis=(1,2,3))
            assert not  np.any(nan_slices), f'{nan_slices.sum()} label slices only contain NANs'

            # check that the labels are always one
            assert np.all(np.sum(y_t.numpy(), axis=-1) == 1)

            nan_frac = np.mean(np.isnan(x_t.numpy()))
            assert nan_frac < 0.01, f'More than 2% nans in the image ({int(nan_frac*100)}%).'

            # check for labels in the slices
            if dimension == 3:
                n_bkr_per_sample = np.sum(y_t.numpy()[...,0], axis=(1,2,3)).astype(int)
                n_object_per_sample = np.sum(y_t.numpy()[...,1], axis=(1,2,3)).astype(int)
            else:
                n_bkr_per_sample = np.sum(y_t.numpy()[...,0], axis=(1,2)).astype(int)
                n_object_per_sample = np.sum(y_t.numpy()[...,1], axis=(1,2)).astype(int)
            if np.all(n_object_per_sample == 0):
                raise Exception('All labels are zero, no objects were found, either the labels are incorrect or there was a problem processing the image.')
        
            n_objects.append(n_object_per_sample)
            n_background.append(n_bkr_per_sample)

        # print(counter)
        counter += 1

    # there should be at least one iteration
    assert counter != 0

    # test that the number of samples per epoch is correct
    if name != 'test':
        assert(counter == cfg.samples_per_volume * cfg.num_files // cfg.batch_size_train)

        # check the fraction of objects per sample
        n_objects = np.array(n_objects)
        n_background = np.array(n_background)

        # get the fraction of samples containing a label
        assert np.mean(n_objects.reshape(-1) > 0) > cfg.percent_of_object_samples / 100

def set_seeds():
    tf.keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

def get_loader(name, module):
    #generate loader
    if name == 'train':
        data_loader = module.SegLoader(name='training_loader')
    elif name == 'vald':
        data_loader = module.SegLoader(
            mode=module.SegLoader.MODES.VALIDATE,
            name='validation_loader'
        )
    elif name == 'test':
        data_loader = module.ApplyLoader(
            mode=module.SegLoader.MODES.APPLY,
            name='test_loader'
        )
    return data_loader

def load_dataset(test_dir):
    # add data path
    file_list = create_test_files.create_test_files(test_dir)

    id_tensor = tf.squeeze(tf.convert_to_tensor(file_list, dtype=tf.string))
    # Create dataset from list of file names
    file_list_ds = tf.data.Dataset.from_tensor_slices(id_tensor)
    # convert it back (for the right types)
    files_list_b = list(file_list_ds.as_numpy_iterator())

    cfg.num_files = len(file_list)

    return file_list, files_list_b

if __name__ == '__main__':
    # run functions for better debugging
    for dimension in [2, 3]:
        for name in ['train', 'vald']:
            for module in [seg_data_loader]:
                test_functions(dimension, name, module)
                test_wrapper(dimension, name, module)