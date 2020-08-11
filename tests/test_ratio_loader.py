import os
import sys
from pathlib import Path

import numpy as np
import pytest


def test_data_loader():

    sys.path.append(os.getcwd())
    from seg_data_loader import SegLoader, SegRatioLoader
    from tests.create_test_files import create_test_files

    data_dir = Path('TestData')
    test_files = np.array([str(data_dir/d.stem.split('-')[1]) for d in data_dir.iterdir() if 'label' in d.name])
    #when no files are there, create them
    if test_files.size == 0:
        create_test_files(data_dir)
        test_files = np.array([str(data_dir/d.stem.split('-')[1]) for d in data_dir.iterdir() if 'label' in d.name])

    ratio_loader = SegRatioLoader(name='training_loader')

    for f in test_files:
        #get samples
        file_bytes = bytearray(f, 'utf-8')
        #get result
        data = ratio_loader._read_file_and_return_numpy_samples(file_bytes)
        [image_obj, labels_obj], [image_bkg, labels_bkg] = data
        #so there are config.samples_per_volume samples, config.percent_of_object_samples
        #have the center on the object, the others on the background

        #get the center
        center_obj = labels_obj[:,labels_obj.shape[1]//2,labels_obj.shape[1]//2]
        center_bkg = labels_bkg[:,labels_bkg.shape[1]//2,labels_bkg.shape[1]//2]
        assert(np.mean(center_obj, axis=0) == pytest.approx([0,1]))
        assert(np.mean(center_bkg, axis=0) == pytest.approx([1,0]))

if __name__ == '__main__':
    test_data_loader()