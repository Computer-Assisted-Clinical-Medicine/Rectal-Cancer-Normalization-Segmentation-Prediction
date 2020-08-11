import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.append(os.getcwd())
from SegmentationNetworkBasis import config as cfg

def create_test_files(test_path):
    shape = (40, 256, 256)
    size = np.prod(shape)
    spacing = np.array([4, 1, 1])
    n_files = 20

    #write random data to file
    training_files = []
    for i in range(n_files):
        patient_number = f'test{i}'
        testpath = test_path #/ patient_number
        if not testpath.exists():
            testpath.mkdir()

        #write labelfile (make a sphere in the center with label one)
        labelfile = testpath / f'{cfg.label_file_name_prefix}{patient_number}.nii'

        pos = (np.indices(shape).T*spacing).T
        center = np.array(shape)*spacing*(0.5 + (np.random.rand(3) - 0.5)*.4)
        #get distance to center
        dist_to_center = np.sqrt(np.sum(np.square(center - pos.T), axis=-1)).T
        dist = (1 - dist_to_center / dist_to_center.max())
        #make the circle a random size
        radius = np.random.rand()*.2 + .05
        labels = dist > 1-radius
        assert(np.sum(labels) > 0)
        label_image = sitk.GetImageFromArray(labels.astype(np.uint8))
        label_image.SetSpacing(np.flip(spacing).tolist())
        sitk.WriteImage(label_image, str(labelfile))

        #write imagefile
        imagefile = testpath / f'{cfg.sample_file_name_prefix}{patient_number}.nii'
        #use sphere plus noise
        image_data = labels*128 + np.abs(np.random.normal(size=shape, scale=128))
        image = sitk.GetImageFromArray(image_data)
        image.SetSpacing(np.flip(spacing).tolist())
        sitk.WriteImage(image, str(imagefile))

        training_files.append(str(testpath))

    return training_files

if __name__ == '__main__':
    #create test data if called
    create_test_files(Path('TestData'))