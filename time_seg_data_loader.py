# %% [markdown]
'''
# Profile Seg_data_loader
## Imports and Definitions

This file is to test the timing of the different data loaders. The functions are
also profiled.
'''

import cProfile
import os
import pstats
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from IPython.display import display

# surpress tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import seg_data_loader
import seg_data_loader_new
import seg_data_loader_no_caching
from SegmentationNetworkBasis import config as cfg
from test_seg_data_loader import (get_loader, load_dataset,
                                  set_parameters_according_to_dimension,
                                  set_seeds)

show_plots = False

def time_functions(dimension, name, module, timing):

    test_dir = Path('test_data')

    set_parameters_according_to_dimension(dimension, 2, test_dir/'data_preprocessed')

    set_seeds()

    #generate loader
    data_loader = get_loader(name, module)

    # get names from csv
    file_list, files_list_b = load_dataset(test_dir)

    # set sampling mode
    if name == 'test':
        cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
    else:
        cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL

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
    print(f'\tExecution time for {name} {dimension}D {module.__name__}: load: {np.mean(load_time):.2f}s, sample: {np.mean(sample_time):.2f}s')
    timing[f'{name}-{dimension}D-{module.__name__}'] = {
        'load file' : np.mean(load_time),
        'get sample' : np.mean(sample_time),
    }

    return timing

def profile_functions(dimension, name, module):

    test_dir = Path('test_data')

    set_parameters_according_to_dimension(dimension, 2, test_dir/'data_preprocessed')

    profile_dir = test_dir / 'profiles'
    if not profile_dir.exists():
        profile_dir.mkdir()
    profile_file = profile_dir / f'{name}-{dimension}D-{module.__name__}.prof'

    set_seeds()

    #generate loader
    data_loader = get_loader(name, module)

    # get names from csv
    file_list, files_list_b = load_dataset(test_dir)

    # set sampling mode
    if name == 'test':
        cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
    else:
        cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL

    def load_all_files():
        for file_id in files_list_b:
            data, lbl = data_loader._load_file(file_id)
            samples, labels = data_loader._get_samples_from_volume(data, lbl)

    # profile the function
    profiler = cProfile.Profile()
    profiler.enable()
    load_all_files()
    profiler.disable()
    # dump stats file
    profiler.dump_stats(profile_file)
    ps = pstats.Stats(profiler)
    ps.sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(15)

    return profile_file

def time_wrapper(dimension, name, module, timing):
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

    # set sampling mode
    if name == 'test':
        cfg.random_sampling_mode = cfg.SAMPLINGMODES.UNIFORM
    else:
        cfg.random_sampling_mode = cfg.SAMPLINGMODES.CONSTRAINED_LABEL

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

    counter = 0
    load_time = []

    start_time = time.perf_counter()

    for sample in dataset:
        if counter == 0:
            setup_time = time.perf_counter() - start_time
        else:
            load_time.append(time.perf_counter() - start_time)
        
        if show_plots:
            if name == 'train':
                if module.__name__ == 'seg_data_loader_new':
                    # convert to numpy
                    x_t, y_t = sample[0].numpy(), sample[1].numpy()
                    plot(dimension, samples_lbl=x_t, labels_lbl=y_t)

        # print(counter)
        counter += 1

        # get time to exclude the checks
        start_time = time.perf_counter()

    assert counter != 0

    if name != 'test':
        assert(counter == cfg.samples_per_volume * cfg.num_files // cfg.batch_size_train)

    if len(load_time) == 0:
        load_time = [setup_time]

    print(f'\tExecution time for one step: {np.mean(load_time):.2f}s ({np.sum(load_time):.2f}s total)')
    print(f'\tSetup time: {setup_time:.2f}s')
    # add to dict
    t_name = f'{name}-{dimension}D-{module.__name__}'
    timing[t_name]['step'] = np.mean(load_time)
    timing[t_name]['setup'] = setup_time
    timing[t_name]['total'] = np.sum(load_time)

    return timing

def plot(dimension, samples_lbl, labels_lbl, samples_bkr=None, labels_bkr=None):
    plt.hist(samples_lbl.reshape(-1))
    plt.title('Histogram of foreground samples')
    plt.show()
    plt.close()

    plt.hist(labels_lbl.reshape(-1))
    plt.title('Histogram of foreground labels')
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
    plt.show()
    plt.close()

# %% [markdown]
'''
## Evaluate the timing
'''

timing = {}

dimensions = [2,3]
names = ['train', 'vald', 'test']
modules = [seg_data_loader_new, seg_data_loader, seg_data_loader_no_caching]

# call functions and time them
for dimension in dimensions:
    for name in names:
        for module in modules:
            print(f'{name} {dimension}D {module.__name__}:')
            time_functions(dimension, name, module, timing)
            time_wrapper(dimension, name, module, timing)

timing_pd = pd.DataFrame(timing).T
# set index
timing_pd.set_index(pd.MultiIndex.from_tuples(tuple(timing_pd.index.str.split('-'))), inplace=True)
display(timing_pd.round(3))

# %% [markdown]
'''
## Analyze the profiles
'''

profile_files = {}

name = 'train'

# call functions and time them
for dimension in dimensions:
    for module in modules:
        print(f'{name} {dimension}D {module.__name__}:')

        # profile the individual functions
        t_name = f'{name}-{dimension}D-{module.__name__}'
        profile_files[t_name] = (profile_functions(dimension, name, module))

print('For a graphical profile call:')
for name, path in profile_files.items():
    print(name)
    print(f'\tsnakeviz {path}')
