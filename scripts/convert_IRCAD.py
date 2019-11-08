import glob
import dicom2nifti
import zipfile
import os
import SimpleITK as sitk
import shutil
from SegmentationNetworkBasis import config as cfg
from BodyContourSegmentation import extract_body_contour

data_path = 'D:\Image_Data\Patient_Data\IRCAD_new\\3Dircadb1'

for f in range(1, 21):

    mask_folder = os.path.join(data_path, "3Dircadb1." + str(f) + "\MASKS_DICOM\MASKS_DICOM")
    if os.path.isdir(mask_folder):
        shutil.rmtree(mask_folder)

    mask_zip = os.path.join(os.path.join(data_path, "3Dircadb1."+str(f)), 'MASKS_DICOM.zip')
    zip_ref = zipfile.ZipFile(mask_zip, 'r')
    zip_ref.extractall(os.path.splitext(mask_zip)[0])
    zip_ref.close()

    img_zip = os.path.join(os.path.join(data_path, "3Dircadb1." + str(f)), 'PATIENT_DICOM.zip')
    zip_ref = zipfile.ZipFile(img_zip, 'r')
    zip_ref.extractall(os.path.splitext(img_zip)[0])
    zip_ref.close()

    im_folder = os.path.join(data_path, "3Dircadb1." + str(f)+"\PATIENT_DICOM\PATIENT_DICOM")

    mask_out = "D:\Image_Data\Patient_Data\IRCAD_new\Data\\" + cfg.label_file_name_prefix + str(f)+".nii"
    im_out = "D:\Image_Data\Patient_Data\IRCAD_new\Data\\" + cfg.sampe_file_name_prefix + str(f)+".nii"

    # convert individual vessel masks
    for arterty in ['artery']:
        vessel_folder = os.path.join(os.path.join(data_path, "3Dircadb1." + str(f) + "\MASKS_DICOM\MASKS_DICOM"), arterty)
        try:
            try:
                dicom2nifti.dicom_series_to_nifti(vessel_folder, vessel_folder + 'art_mask.nii')
            except OSError as err:
                if err.errno == 2:
                    raise err
            print("    Wrote individual vessel file", arterty)
        except:
            print("    Failed to convert individual vessel mask for " + arterty)

    # convert individual vessel masks
    for vein in ['portalvein', 'venacava', 'venoussystem']:
        vessel_folder = os.path.join(os.path.join(data_path, "3Dircadb1." + str(f) + "\MASKS_DICOM\MASKS_DICOM"),
                                     vein)
        try:
            try:
                dicom2nifti.dicom_series_to_nifti(vessel_folder, vessel_folder + 'vein_mask.nii')
            except OSError as err:
                if err.errno == 2:
                    raise err
            print("    Wrote individual vessel file", vein)
        except:
            print("    Failed to convert individual vessel mask for " + vein)

    # convert image
    try:
        try:
            dicom2nifti.dicom_series_to_nifti(im_folder, im_out)
        except OSError:
            pass
        data_img = sitk.ReadImage(im_out)
        body_mask = extract_body_contour.get_mask(data_img, 'ct')
        masked_img = sitk.Mask(data_img, body_mask, -1000)
        sitk.WriteImage(masked_img, im_out)
        print("  Image done")
    except Exception as err:
        print("  Couldn't convert Image " + im_folder, err)

    # fuse vessel labels
    data_img = sitk.ReadImage(im_out)
    vessel_mask_combined = sitk.Image(data_img.GetSize(), sitk.sitkUInt8)
    vessel_mask_combined.CopyInformation(data_img)
    arterty_vessel_search_path = os.path.join(os.path.join(data_path, '3Dircadb1.' + str(f) + '\MASKS_DICOM\MASKS_DICOM'),
                                     '*art_mask.nii')
    vein_vessel_search_path = os.path.join(os.path.join(data_path, '3Dircadb1.' + str(f) + '\MASKS_DICOM\MASKS_DICOM'),
                                           '*vein_mask.nii')
    try:
        for indiv_vessel_mask in glob.glob(vein_vessel_search_path):
            vessel_mask = sitk.ReadImage(indiv_vessel_mask)
            vessel_mask = sitk.Cast(vessel_mask, sitk.sitkUInt8)
            vessel_mask_combined = sitk.Mask(vessel_mask_combined, vessel_mask < 1, 2)
            print('    Fused ' + os.path.basename(indiv_vessel_mask))
        for indiv_vessel_mask in glob.glob(arterty_vessel_search_path):
            vessel_mask = sitk.ReadImage(indiv_vessel_mask)
            vessel_mask_combined = sitk.Mask(vessel_mask_combined, vessel_mask < 1, 1)
            vessel_mask = sitk.Cast(vessel_mask, sitk.sitkUInt8)
            print('    Fused ' + os.path.basename(indiv_vessel_mask))

        vessel_mask_combined = sitk.Cast(vessel_mask_combined, sitk.sitkUInt8)
        sitk.WriteImage(vessel_mask_combined, mask_out)
        print("  Label done")
    except RuntimeError as err:
        print("Fusion failed in " + arterty_vessel_search_path, vein_vessel_search_path, err)

    print("Converted data set " + str(f))
    print("-------------------------------------------")
