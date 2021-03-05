import os
from pathlib import Path

import GPUtil
import numpy as np
import pytest
import SimpleITK as sitk
import tensorflow as tf

import create_test_files
import seg_data_loader
from SegmentationNetworkBasis import config as cfg

# TODO: add tests for apply loader

def set_parameters_according_to_dimension(dim, num_channels, preprocessed_dir):
    """This function will set up the shapes in the cfg module so that they
    will run on the current GPU.
    """
    
    cfg.num_files_vald = 2

    cfg.num_channels = num_channels
    cfg.train_dim = 128 # the resolution in plane
    cfg.num_slices_train = 32 # the resolution in z-direction

    # determine batch size
    cfg.batch_size_train = estimate_batch_size(dim)
    cfg.batch_size_valid = cfg.batch_size_train

    # set shape according to the dimension
    if dim == 2:
        # set shape
        cfg.train_input_shape = [cfg.train_dim, cfg.train_dim, cfg.num_channels]
        cfg.train_label_shape = [cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]

        # set sample numbers
        # there are 10-30 layers per image containing foreground data. Half the
        # samples are taken from the foreground, so take about 64 samples
        # to cover all the foreground pixels at least once on average, but
        cfg.samples_per_volume = 64
        cfg.batch_capacity_train = 4*cfg.samples_per_volume # chosen as multiple of samples per volume

    elif dim == 3:
        # set shape
        # if batch size too small, decrease z-extent
        if cfg.batch_size_train < 4:
            cfg.num_slices_train = cfg.num_slices_train // 2
            cfg.batch_size_train = cfg.batch_size_train * 2
            # if still to small, decrease patch extent in plane
            if cfg.batch_size_train < 4:
                cfg.train_dim = cfg.train_dim // 2
                cfg.batch_size_train = cfg.batch_size_train * 2
        cfg.train_input_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_channels]
        cfg.train_label_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
        
        # set sample numbers
        # most patches should cover the whole tumore, so a lower sample number
        # can be used
        cfg.samples_per_volume = 8
        cfg.batch_capacity_train = 4*cfg.samples_per_volume # chosen as multiple of samples per volume
        
    # see if the batch size is bigger than the validation set
    if cfg.samples_per_volume * cfg.num_files_vald < cfg.batch_size_valid:
        cfg.batch_size_valid = cfg.samples_per_volume * cfg.num_files_vald
    else:
        cfg.batch_size_valid = cfg.batch_capacity_train

    # set config
    if not preprocessed_dir.exists():
        preprocessed_dir.mkdir(parents=True)
    cfg.preprocessed_dir = str(preprocessed_dir)
    cfg.normalizing_method == cfg.NORMALIZING.PERCENT5


def estimate_batch_size(dim):
    """The batch size estimation is basically trail and error. So far tested
    with 128x128x2 patches in 2D and 128x128x32x2 in 3D, if using different
    values, guesstimate the relation to the memory.

    Returns
    -------
    int
        The recommended batch size
    """
    # set batch size
    # determine GPU memory (in MB)
    gpu_number = int(tf.test.gpu_device_name()[-1])
    gpu_memory = int(np.round(GPUtil.getGPUs()[gpu_number].memoryTotal))

    a_name = 'UNet' # name is irrelevant

    if a_name == 'UNet':
        # filters scale after the first filter, so use that for estimation
        first_f = 8
        if dim == 2:
            # this was determined by trail and error for 128x128x2 patches
            memory_consumption_guess = 2 * first_f
        elif dim == 3:
            # this was determined by trail and error for 128x128x32x2 patches
            memory_consumption_guess = 64 * first_f
    else:
        raise NotImplementedError('No heuristic implemented for this network.')
    
    # return estimated recommended batch number
    return np.round(gpu_memory // memory_consumption_guess)

@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('name', ['train', 'vald'])
@pytest.mark.parametrize('module', [seg_data_loader])
def test_functions(dimension, name, module):
    """Test the individual functions contained in the wrapper.

    Parameters
    ----------
    dimension : int
        The dimension (2 or 3)
    name : str
        The name, train, test or vald
    module : DataLoader
        the data loader to use]
    """

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
    """Test the complete wrapper and check shapes

    Parameters
    ----------
    dimension : int
        The dimension (2 or 3)
    name : str
        The name, train, test or vald
    module : DataLoader
        the data loader to use

    Raises
    ------
    Exception
        Error as detected
    """
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