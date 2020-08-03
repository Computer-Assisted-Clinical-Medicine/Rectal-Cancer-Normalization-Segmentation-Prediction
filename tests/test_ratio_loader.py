import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk


def import_packages():
    sys.path.append(os.getcwd())
    from seg_data_loader import SegLoader, SegRatioLoader
    from SegmentationNetworkBasis import config as cfg

@pytest.fixture
def create_test_files(test_path):
    shape = (64, 256, 256)
    size = np.prod(shape)
    spacing = np.array([4, 1, 1])
    n_files = 20

    #write random data to file
    training_files = []
    for i in range(n_files):
        testpath = test_path / f'test{i}'
        testpath.mkdir()

        #write labelfile (make a sphere in the center with label one)
        labelfile = testpath / f'labels.nii.gz'

        pos = (np.indices((32,256,256)).T*spacing).T
        center = np.array(shape)*spacing*(0.5 + (np.random.rand(3) - 0.5)*.25)
        #get distance to center
        dist_to_center = np.sqrt(np.sum(np.square(center - pos.T), axis=-1)).T
        dist = (1 - dist_to_center / dist_to_center.max())
        #make the circle a random size
        labels = dist > (np.random.rand()*.4 + .6)
        label_image = sitk.GetImageFromArray(labels.astype(np.uint8))
        label_image.SetSpacing(np.flip(spacing).tolist())
        sitk.WriteImage(label_image, str(labelfile))

        #write imagefile
        imagefile = testpath / f'image.nii.gz'
        #use sphere plus noise
        image_data = labels*256 + np.abs(np.random.normal(size=shape, scale=30))
        image = sitk.GetImageFromArray(image_data)
        image.SetSpacing(np.flip(spacing).tolist())
        sitk.WriteImage(image, str(imagefile))

        training_files.append(str(testpath))

    return training_files

def test_data_loader():

    data_dir = Path('TestData')
    test_files = np.array([str(d) for d in data_dir.iterdir() if d.is_dir()])

    sys.path.append(os.getcwd())
    from seg_data_loader import SegLoader, SegRatioLoader
    from SegmentationNetworkBasis import config as cfg

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
