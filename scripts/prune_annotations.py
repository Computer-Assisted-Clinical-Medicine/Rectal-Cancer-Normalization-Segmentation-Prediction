import SimpleITK as sitk
import numpy as np


def filter_by_size(ccm, n_components, min_size):
    new_index = 1
    for c in range(new_index, n_components+1):
        mask = ccm == c
        count = np.sum(mask)
        # print(c, count)
        if count < min_size:
            ccm[mask] = 0
            print('  Removed Component ', c)
        else:
            if new_index < c:
                ccm[mask] = new_index
                # print('  Renamed Component ', c, 'to', new_index)

            new_index += 1

    print(new_index-1, ' Components remaining.')

    assert len(np.unique(ccm)) == new_index
    return ccm > 0


def extract_largest_connected_component_sitk(segmentation):
    connected_component_filter = sitk.ConnectedComponentImageFilter()
    connected_component_filter.SetFullyConnected(True)
    component_map = sitk.GetArrayFromImage(connected_component_filter.Execute(segmentation))
    print(connected_component_filter.GetObjectCount())
    new_map = np.zeros(shape=component_map.shape)
    binc = np.bincount(component_map.flat)
    # try:
    #     cc_id = np.argmax(binc[1:])+1
    #     largestCC = component_map == cc_id
    #     new_map[largestCC] = 1
    # except (RuntimeError, ValueError) as err:
    #     print('Empty Volume: ', err)
    new_map = filter_by_size(component_map, connected_component_filter.GetObjectCount(), 5)
    component_image = sitk.GetImageFromArray(new_map.astype(np.uint8))
    component_image.CopyInformation(segmentation)
    return component_image

for i in range(1, 21):
    label = sitk.ReadImage(r'D:\Image_Data\Patient_Data\IRCAD_new\Data\vessel-segmentation_revised-' + str(i) + '.nii')
    print('Case ', str(i))
    pred_img_art = extract_largest_connected_component_sitk(label == 1)
    pred_img_vein = extract_largest_connected_component_sitk(label == 2)
    connected_img = sitk.Mask(pred_img_art, pred_img_vein < 1, 2)
    sitk.WriteImage(connected_img, r'D:\Image_Data\Patient_Data\IRCAD_new\Data\vessel-segmentation_revised_connected-' + str(i) + '.nii')