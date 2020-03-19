import SimpleITK as sitk

# image = sitk.ReadImage(r'T:\IRCAD\Data\ct-volume-19_old.nii')
# label = sitk.ReadImage(r'T:\IRCAD\Data\vessel-segmentation_revised-19_old.nii')
#
# cropped_image = sitk.Crop(image, [0, 0, 0], [0, 0, 45])
# cropped_label = sitk.Crop(label, [0, 0, 0], [0, 0, 45])
#
# sitk.WriteImage(cropped_image, r'T:\IRCAD\Data\ct-volume-19.nii')
# sitk.WriteImage(cropped_label, r'T:\IRCAD\Data\vessel-segmentation_revised-19.nii')
#
#
# image = sitk.ReadImage(r'T:\IRCAD\Data\ct-volume-20_old.nii')
# label = sitk.ReadImage(r'T:\IRCAD\Data\vessel-segmentation_revised-20_old.nii')
#
# cropped_image = sitk.Crop(image, [0, 0, 90], [0, 0, 0])
# cropped_label = sitk.Crop(label, [0, 0, 90], [0, 0, 0])
#
# sitk.WriteImage(cropped_image, r'T:\IRCAD\Data\ct-volume-20.nii')
# sitk.WriteImage(cropped_label, r'T:\IRCAD\Data\vessel-segmentation_revised-20.nii')

import SegmentationNetworkBasis.NetworkBasis.image as image
label = sitk.ReadImage(r'T:\IRCAD\Data\vessel-segmentation_revised-19.nii')
pred_img_art = image.extract_largest_connected_component_sitk(label == 1)
pred_img_vein = image.extract_largest_connected_component_sitk(label == 2)
connected_img = sitk.Mask(pred_img_art, pred_img_vein < 1, 2)
sitk.WriteImage(connected_img, r'T:\IRCAD\Data\vessel-segmentation_revised_connected-20.nii')
pass