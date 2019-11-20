import glob
import dicom2nifti
import zipfile
import os
import numpy as np
import SimpleITK as sitk
import shutil
from SegmentationNetworkBasis import config as cfg
from BodyContourSegmentation import extract_body_contour

np.random.seed(42)

data_path = 'D:\Image_Data\\Numerical_Phantoms\\XCAT\\NRRD_Breathe'

models = [77, 80, 92, 93, 96, 108, 118, 128, 139, 141, 144, 145, 146, 148, 150, 154, 155, 157, 159, 163, 164, 167, 169,
          171, 173, 178, 180, 184, 196, 200, 201, 168, 71, 76, 86, 89, 98, 99, 106, 117, 140, 142, 143, 147, 149, 151,
          152, 153, 162, 166, 175, 176, 182, 170, 401, 447]
energies = [90, 100, 110, 120]
points = [1, 2, 3, 4, 5]

for i in range(len(models)):

    if i % 2 == 0:
        selected_energies = np.random.choice(energies, 2, False)
    else:
        selected_energies = np.setdiff1d(energies, selected_energies)

    selected_points = np.random.choice(points, 2, False)

    for e, p in zip(selected_energies, selected_points):

        orig_label = os.path.join(data_path, 'Model' + str(models[i]) + '_Energy' + str(e) + '_act_' + str(p) + '.nrrd')
        out_label = os.path.join(data_path, 'Data', cfg.label_file_name_prefix + str(models[i]) + str(e) + str(p) + ".nii")
        out_img = os.path.join(data_path, 'Data', cfg.sample_file_name_prefix + str(models[i]) + str(e) + str(p) + ".nii")
        orig_img = os.path.join(data_path, 'Model' + str(models[i]) + '_Energy' + str(e) + '_atn_' + str(p) + '.nrrd')

        # convert image
        try:
            data_img = sitk.ReadImage(orig_img)
            sitk.WriteImage(data_img, out_img)
            print("  Image done")
        except Exception as err:
            print("  Couldn't convert Image " + str(models[i]) + str(e) + str(p), err)

        # convert label
        try:
            label_img = sitk.ReadImage(orig_label)
            sitk.WriteImage(label_img, out_label)
            print("  Label done")
        except Exception as err:
            print("  Couldn't convert Image " + str(models[i]) + str(e) + str(p), err)


        print("Converted data set " + str(models[i]) + str(e) + str(p))
        print("-------------------------------------------")
