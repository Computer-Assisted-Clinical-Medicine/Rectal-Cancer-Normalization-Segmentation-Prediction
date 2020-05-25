import SimpleITK as sitk
import numpy as np


def get_major_vessel_mask(segmentation):
    eroded = sitk.BinaryErode(segmentation, 5)
    dilated = sitk.BinaryDilate(eroded, 10)
    return dilated


for i in range(1, 21):
    label = sitk.ReadImage(r'D:\Image_Data\Patient_Data\IRCAD_new\Data\vessel-segmentation_revised-' + str(i) + '.nii')
    print('Case ', str(i))
    label_img_art = label == 1
    label_img_vein = label == 2

    major_vessel_mask_art = get_major_vessel_mask(label_img_art)
    major_vessel_mask_vein = get_major_vessel_mask(label_img_vein)

    vessel_mask_combined = sitk.MaskNegated(major_vessel_mask_art, major_vessel_mask_vein, 2)

    sitk.WriteImage(vessel_mask_combined, r'D:\Image_Data\Patient_Data\IRCAD_new\Data\vessel-segmentation_revised-major-' + str(i) + '.nii')