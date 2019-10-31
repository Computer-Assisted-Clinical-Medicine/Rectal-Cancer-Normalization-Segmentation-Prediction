import glob
import dicom2nifti
import zipfile
import os
import SimpleITK as sitk
import shutil
import numpy as np
import pandas as pd
from SegmentationNetworkBasis import config as cfg
from BodyContourSegmentation import extract_body_contour

data_path = 'T:\DVN-Synth'

for f in range(1, 137):  # Test Data does not have vessel annotations
    orig_label = os.path.join(data_path, 'seg', str(f) +".nii.gz")
    out_label = os.path.join(data_path, 'Data', cfg.label_file_name_prefix + str(f)+".nii")
    out_img = os.path.join(data_path, 'Data', cfg.sampe_file_name_prefix + str(f)+".nii")
    orig_img = os.path.join(data_path, 'raw', str(f) +".nii.gz")

    # convert image
    try:
        data_img = sitk.ReadImage(orig_img)
        sitk.WriteImage(data_img, out_img)
        print("  Image done " + str(f))
    except (NameError, RuntimeError, IndexError) as err:
        print("  Couldn't convert Image " + str(f), err)
        continue

    try:
        data_img = sitk.ReadImage(out_img)
        label_img = sitk.ReadImage(orig_label)
        label_img.CopyInformation(data_img)
        sitk.WriteImage(label_img, out_label)
        print("  Label done " + str(f))
    except RuntimeError as err:
        print("Fusion failed in " + str(f), err)
        pass

print("-------------------------------------------")
