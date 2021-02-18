# this file is for debugging, it runs the data loader without tensorflow
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from seg_data_loader import SegLoader, SegRatioLoader
from SegmentationNetworkBasis import config as cfg

show_plots = False

def set_parameters_according_to_dimension(dimension, num_channels):
    if dimension == 2:
        cfg.num_channels = num_channels
        #cfg.train_dim = 256
        cfg.samples_per_volume = 160
        cfg.batch_capacity_train = 750
        cfg.batch_capacity_valid = 450
        cfg.train_input_shape = [cfg.train_dim, cfg.train_dim, cfg.num_channels]
        cfg.train_label_shape = [cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
        #cfg.test_dim = 512
        cfg.test_data_shape = [cfg.test_dim, cfg.test_dim, cfg.num_channels]
        cfg.test_label_shape = [cfg.test_dim, cfg.test_dim, cfg.num_classes_seg]
        cfg.batch_size_train = 64
        cfg.batch_size_test = 1
    elif dimension == 3:
        cfg.num_channels = num_channels
        #cfg.train_dim = 128
        cfg.samples_per_volume = 80
        cfg.batch_capacity_train = 250
        cfg.batch_capacity_valid = 150
        cfg.num_slices_train = 8
        cfg.train_input_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_channels]
        cfg.train_label_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
        #cfg.test_dim = 512
        cfg.num_slices_test = 32
        cfg.test_data_shape = [cfg.num_slices_test, cfg.test_dim, cfg.test_dim, cfg.num_channels]
        cfg.test_label_shape = [cfg.num_slices_test, cfg.test_dim, cfg.test_dim, cfg.num_classes_seg]
        cfg.batch_size_train = 4 #Otherwise, VNet 3D fails
        cfg.batch_size_test = 1

# remember timing
timing = {}

for dimension in [2, 3]:

    n_epochs = 1

    print(f'{dimension}D Data:')

    set_parameters_according_to_dimension(dimension, 2)

    data_dir = Path(os.environ['data_dir'])
    experiment_dir = Path(os.environ['experiment_dir'])
    if not experiment_dir.exists():
        experiment_dir.mkdir()

    # get names from csv
    train_list = np.loadtxt(data_dir / 'train_IDs.csv', dtype='str').reshape((-1))
    # add data path
    file_list = np.array([str(data_dir / t) for t in train_list])

    cfg.num_files = len(file_list)

    tf.keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    #generate loader
    print('Initialize Loaders')
    data_loader_train = SegRatioLoader(name='training_loader')
    print('Load validation Dataset')
    validation_dataset = SegRatioLoader(
        mode=SegRatioLoader.MODES.VALIDATE,
        name='validation_loader'
    )
    print('Load test dataset')
    testloader = SegLoader(
        mode=SegLoader.MODES.APPLY,
        name='test_loader'
    )

    id_tensor = tf.squeeze(tf.convert_to_tensor(file_list, dtype=tf.string))
    # Create dataset from list of file names
    file_list_ds = tf.data.Dataset.from_tensor_slices(id_tensor)
    # convert it back (for the right types)
    files_list_b = list(file_list_ds.as_numpy_iterator())

    # print('This is the file list as tensor slices:')
    # print('\n'.join([str(s) for s in files_list_b]))

    data_file, label_file = testloader._get_filenames(str(file_list[0]))
    first_image = sitk.GetArrayFromImage(sitk.ReadImage(data_file))

    for data_loader, name in zip([data_loader_train, validation_dataset, testloader], ['train', 'val', 'test']):
        print(f'Loading Dataset {name}.')

        # set sampling mode
        if name == 'test':
            cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
        else:
            cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL

        print('\tLoad a numpy sample')
        data_read = data_loader._read_file_and_return_numpy_samples(files_list_b[0])
        if name == 'test':
            samples_lbl = data_read[0]
        else:
            ((samples_lbl, labels_lbl), (samples_bkr, labels_bkr)) = data_read
            print(f'\tSamples from foreground shape: {samples_lbl.shape}')
            print(f'\tLabels from foreground shape: {labels_lbl.shape}')
            print(f'\tSamples from background shape: {samples_bkr.shape}')
            print(f'\tLabels from background shape: {labels_bkr.shape}')

            assert(samples_lbl.shape[:-1] == labels_lbl.shape[:-1])
            assert(samples_bkr.shape[:-1] == labels_bkr.shape[:-1])

            nan_slices = np.all(np.isnan(samples_lbl), axis=(1,2,3))
            if np.any(nan_slices):
                print(f'{nan_slices.sum()} foreground sample slices only contain NANs')

            nan_slices = np.all(np.isnan(labels_lbl), axis=(1,2,3))
            if np.any(nan_slices):
                print(f'{nan_slices.sum()} foreground label slices only contain NANs')

            nan_slices = np.all(np.isnan(samples_bkr), axis=(1,2,3))
            if np.any(nan_slices):
                print(f'{nan_slices.sum()} background sample slices only contain NANs')

            nan_slices = np.all(np.isnan(labels_bkr), axis=(1,2,3))
            if np.any(nan_slices):
                print(f'{nan_slices.sum()} background label slices only contain NANs')

            plt.hist(samples_lbl.reshape(-1))
            plt.title('Histogram of foreground samples')
            if show_plots:
                plt.show()
            plt.close()

            plt.hist(labels_lbl.reshape(-1))
            plt.title('Histogram of foreground labels')
            if show_plots:
                plt.show()
            plt.close()

            plt.hist(samples_bkr.reshape(-1))
            plt.title('Histogram of background samples')
            if show_plots:
                plt.show()
            plt.close()

            plt.hist(labels_bkr.reshape(-1))
            plt.title('Histogram of background labels')
            if show_plots:
                plt.show()
            plt.close()

            nsamples = 5
            ncols = samples_lbl.shape[-1] + 1
            nrows = nsamples
            indices = np.sort(np.random.choice(np.arange(samples_lbl.shape[0]), nsamples))
            fig, axes = plt.subplots(nrows, ncols, figsize=(11,9))

            for ax_r, sample_r, label in zip(axes, samples_lbl[indices], labels_lbl[indices]):
                index = np.random.choice(np.arange(sample_r.shape[0]))
                for ax, sample in zip(ax_r[:-1], np.moveaxis(sample_r, -1 ,0)):
                    if dimension == 3:
                        ax.imshow(sample[index])
                    else:
                        ax.imshow(sample)
                if dimension == 3:
                    ax_r[-1].imshow(label[index,...,:-1], vmin=0, vmax=1)
                else:
                    ax_r[-1].imshow(label[...,:-1], vmin=0, vmax=1)

            plt.tight_layout()
            if show_plots:
                plt.show()
            plt.close()

        # call the wrapper function
        data_loader._read_wrapper(id_data_set=tf.squeeze(tf.convert_to_tensor(file_list[0], dtype=tf.string)))

        # time the individual functions
        load_time = []
        sample_time = []
        for file_id in files_list_b:
            start_time = time.perf_counter()
            data, lbl = data_loader._load_file(file_id)
            load = time.perf_counter()
            samples, labels = data_loader._get_samples_from_volume(data, lbl)
            finished = time.perf_counter()
            # save times
            load_time.append(load - start_time)
            sample_time.append(finished - load)
        print(f'\tExecution time for {name} {dimension}D: load: {np.mean(load_time):.2f}s, sample: {np.mean(sample_time):.2f}s')
        timing[f'{name}-{dimension}D'] = {
            'load' : np.mean(load_time),
            'sample' : np.mean(sample_time),
        }

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
                batch_size=cfg.batch_size_train,
                read_threads=cfg.vald_reader_instances
            )
        else:
            dataset = data_loader(
                file_list,
                batch_size=cfg.batch_size_train,
                read_threads=cfg.vald_reader_instances
            )

        print('\tLoad samples using the data loader')

        counter = 0
        load_time = []

        start_time = time.perf_counter()

        for sample in dataset:
            # test set only contains samples, not labels
            if name == 'test':
                x_t = sample
            else:
                x_t, y_t = sample
            if counter == 0:
                setup_time = time.perf_counter() - start_time
            else:
                load_time.append(time.perf_counter() - start_time)

            # check shape
            if name != 'test':
                assert(cfg.batch_size_train == x_t.shape[0])
                assert(cfg.num_channels == x_t.shape[-1])
                assert(cfg.batch_size_train == y_t.shape[0])
            else:
                assert(first_image.shape[0] == x_t.numpy().shape[0])

            # look for nans
            nan_slices = np.all(np.isnan(x_t.numpy()), axis=(1,2,3))
            if np.any(nan_slices):
                print(f'{nan_slices.sum()} sample slices only contain NANs')

            if name != 'test':
                nan_slices = np.all(np.isnan(y_t.numpy()), axis=(1,2,3))
                if np.any(nan_slices):
                    print(f'{nan_slices.sum()} label slices only contain NANs')

            nan_frac = np.mean(np.isnan(x_t.numpy()))
            if nan_frac > 0.02:
                print(f'More than 2% nans in the image ({int(nan_frac*100)}%).') 

            # print(f'\t\tShape x:{x_t.shape}, Shape y:{y_t.shape}')

            # check for labels in the slices
            if name != 'test':
                if dimension == 3:
                    n_object_per_slice = np.sum(y_t.numpy(), axis=(1,2,3,4)).astype(int)
                else:
                    n_object_per_slice = np.sum(y_t.numpy(), axis=(1,2,3)).astype(int)
                if np.any(n_object_per_slice == 0):
                    raise Exception('All labels are zero, no objects were found, either the labels are incorrect or there was a problem processing the image.')
            
            # print(counter)
            counter += 1

            # get time to exclude the checks
            start_time = time.perf_counter()

        if name != 'test':
            assert(counter == cfg.samples_per_volume * cfg.num_files // cfg.batch_size_train)
        else:
            assert(counter == cfg.batch_size_test)

        if len(load_time) == 0:
            load_time = [setup_time]

        print(f'Execution time for one step for {name} {dimension}D: {np.mean(load_time):.2f}s ({np.sum(load_time):.2f}s total)')
        print(f'Setup time for {name} {dimension}D: {setup_time:.2f}s')
        # add to dict
        t_name = f'{name}-{dimension}D'
        timing[t_name]['step'] = np.mean(load_time)
        timing[t_name]['total'] = np.sum(load_time)
        timing[t_name]['setup'] = setup_time

        print('finished')

        plt.hist(load_time)
        plt.show()
        plt.close()

for name, times in timing.items():
    print(name)
    line = ''
    for t_name, value in times.items():
        line += f'\t{t_name:5} : {value:2.2f}s '
    print(line)
