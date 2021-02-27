import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.append(os.getcwd())
from SegmentationNetworkBasis import config as cfg

def create_test_files(test_path=Path('test_data'), n_files = 20):
    spacing = np.array([4, 1, 1, 1])

    if not test_path.exists():
        test_path.mkdir()

    #write random data to file
    training_files = []
    for i in range(n_files):
        patient_number = f'test{i}'

        labelfile = test_path / f'{cfg.label_file_name_prefix}{patient_number}.nrrd'
        imagefile = test_path / f'{cfg.sample_file_name_prefix}{patient_number}.nrrd'

        training_files.append(str(test_path / f'{patient_number}'))

        if labelfile.exists() and imagefile.exists():
            continue
        
        # take a random number of slices
        shape = (np.random.randint(low=20, high=60), 256, 256, cfg.num_channels)
        size = np.prod(shape)

        pos = (np.indices(shape[:3]).T*spacing[:3]).T
        center = np.array(shape[:3])*spacing[:3]*(0.5 + (np.random.rand(3) - 0.5)*.4)
        #get distance to center (ignoring the channel)
        dist_to_center = np.sqrt(np.sum(np.square(center - pos[:3].T), axis=-1)).T
        dist = (1 - dist_to_center / dist_to_center.max())
        #make the circle a random size
        radius = np.random.rand()*.2 + .05
        labels = dist > 1-radius
        assert(np.sum(labels) > 0)
        label_image = sitk.GetImageFromArray(labels.astype(np.uint8))
        label_image = sitk.Cast(label_image, sitk.sitkUInt8)
        label_image.SetSpacing(np.flip(spacing).tolist())
        #write labelfile (make a sphere in the center with label one)
        sitk.WriteImage(label_image, str(labelfile))

        #use sphere
        image_data = np.repeat(np.expand_dims(labels, axis=3), repeats=2, axis=3)*128
        #add noise
        image_data = image_data + np.abs(np.random.normal(size=shape, scale=128))
        image = sitk.GetImageFromArray(image_data)
        image.SetSpacing(np.flip(spacing).tolist())
        #write imagefile
        sitk.WriteImage(image, str(imagefile))

    return training_files

if __name__ == '__main__':
    #create test data if called
    create_test_files(Path('test_data'))