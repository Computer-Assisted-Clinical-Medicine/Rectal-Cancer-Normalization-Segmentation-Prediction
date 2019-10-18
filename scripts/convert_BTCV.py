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

data_path = 'T:\BTCV'

cropping_info = pd.read_csv(os.path.join(data_path, 'cropping.csv'), dtype=object).as_matrix()

for f in range(1, 41):  # Test Data does not have vessel annotations
    orig_label = os.path.join(data_path, 'label_btcv_multiorgan', "label" + "%04d" % f +".nii.gz")
    out_label = os.path.join(data_path, 'Data', cfg.label_file_name_prefix + str(f)+".nii")
    out_img = os.path.join(data_path, 'Data', cfg.sampe_file_name_prefix + str(f)+".nii")
    if f < 41:
        orig_img = os.path.join(data_path, 'RawData', 'Training', 'img', "img" + "%04d" % f + ".nii.gz")
    else:
        orig_img = os.path.join(data_path, 'RawData', 'Testing', 'img', "img" + "%04d" % f + ".nii.gz")

    crop_index = np.where(cropping_info[43:, 2] == "%04d" % f)

    # convert image
    try:
        data_img = sitk.ReadImage(orig_img)
        data_img = data_img[:, :, int(cropping_info[43 + crop_index[0], 7][0]):int(cropping_info[43 + crop_index[0], 8][0])]
        print(cropping_info[43 + crop_index[0], :])
        body_mask = extract_body_contour.get_mask(data_img, 'ct')
        masked_img = sitk.Mask(data_img, body_mask, -1000)
        img_array = sitk.GetArrayFromImage(masked_img)
        img_array = np.flip(img_array, axis=1)
        new_img_data = sitk.GetImageFromArray(img_array)
        new_img_data.CopyInformation(masked_img)
        sitk.WriteImage(new_img_data, out_img)
        print("  Image done " + str(f))
    except (NameError, RuntimeError, IndexError):
        # print("  Couldn't convert Image " + str(f), err)
        continue
    # fuse vessel labels
    try:
        data_img = sitk.ReadImage(out_img)
        vessel_mask_combined = sitk.Image(data_img.GetSize(), sitk.sitkUInt8)
        vessel_mask_combined.CopyInformation(data_img)

        label_img = sitk.ReadImage(orig_label)
        label_img = label_img[:, :,
                   int(cropping_info[43 + crop_index[0], 7][0]):int(cropping_info[43 + crop_index[0], 8][0])]
        vessel_mask_combined = sitk.MaskNegated(vessel_mask_combined, label_img == 9, 2)
        vessel_mask_combined = sitk.MaskNegated(vessel_mask_combined, label_img == 10, 2)
        print('    Fused veins')
        vessel_mask_combined = sitk.MaskNegated(vessel_mask_combined, label_img == 8, 1)
        print('    Fused arteries')
        lbl_array = sitk.GetArrayFromImage(vessel_mask_combined)
        lbl_array = np.flip(lbl_array, axis=1)
        new_lbl_data = sitk.GetImageFromArray(lbl_array)
        new_lbl_data.CopyInformation(vessel_mask_combined)
        sitk.WriteImage(new_lbl_data, out_label)
        print("  Label done " + str(f))
    except RuntimeError as err:
        # print("Fusion failed in " + str(f), err)
        pass

print("-------------------------------------------")
