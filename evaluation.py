import math
import SegmentationNetworkBasis.NetworkBasis.metric as Metric
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import os
from SegmentationNetworkBasis.segbasisloader import SegBasisLoader
from liversegloader import LiverSegLoader
import SegmentationNetworkBasis.NetworkBasis.image as Image
from SegmentationNetworkBasis import config as cfg


def make_csv_header():
    if cfg.organ == cfg.ORGANS.LIVERANDTUMORS:
        header_row = ['File Number', 'Slices', 'Liver Volume (L)', 'Liver Volume (P)', 'Lesion Volume (L)', 'Lesion Volume (P)',
                      'Dice - Liver', 'Volume Similarity - Liver', 'False Negative - Liver', 'False Positive - Liver',
                      'Dice - Lesion', 'Volume Similarity - Lesion', 'False Negative - Lesion', 'False Positive - Lesion',
                      'Hausdorff - Liver', 'Mean Surface Distance - Liver', 'Median Surface Distance - Liver',
                      'STD Surface Distance - Liver', 'Max Surface Distance - Liver', 'Time']
    elif cfg.organ == cfg.ORGANS.LIVER:
        header_row = ['File Number', 'Slices', cfg.organ.value + ' Volume (L)', cfg.organ.value + ' Volume (P)',
                  'Dice - ' + cfg.organ.value, 'Volume Similarity - ' + cfg.organ.value, 'False Negative - ' + cfg.organ.value,
                  'False Positive - ' + cfg.organ.value, 'Hausdorff - ' + cfg.organ.value, 'Mean Symmetric Surface Distance - ' + cfg.organ.value,
                  'Median Symmetric Surface Distance - ' + cfg.organ.value, 'STD Symmetric Surface Distance - ' + cfg.organ.value,
                  'Max Symmetric Surface Distance - ' + cfg.organ.value, 'Time']
    return header_row


@staticmethod
def get_organ_mask(label, organ):
    if organ == cfg.ORGANS.LIVER:
        label = label > 0  # only liver
    elif organ == cfg.ORGANS.TUMORS:
        label = label > 1  # only tumors
    elif organ == cfg.ORGANS.LIVERANDTUMORS:
        pass
    else:
        raise ValueError(organ + ' is not a valid organ!')
    return label


def evaluate_segmentation_prediction(pred_img, result_metrics, data_info, path):
    path = path[0]
    folder, file_number = os.path.split(path)

    # load label for evaluation
    Label = sitk.ReadImage(os.path.join(folder, ('segmentation-' + file_number + '.nii')))
    if cfg.organ == cfg.ORGANS.LIVER or cfg.organ == cfg.ORGANS.TUMORS:
        Label = LiverSegLoader.get_organ_mask(Label, cfg.organ)

        # This is necessary as in some data sets this is incorrect.
        Label.SetDirection(data_info['orig_direction'])
        Label.SetOrigin(data_info['orig_origin'])
        Label.SetSpacing(data_info['orig_spacing'])

        result_metrics['Slices'] = data_info['orig_size'][2]
        result_metrics[cfg.organ.value + ' Volume (L)'] = Metric.get_ml_sitk(Label)
        result_metrics[cfg.organ.value + ' Volume (P)'] = Metric.get_ml_sitk(pred_img)

        orig_dice, orig_vs, orig_fn, orig_fp = Metric.overlap_measures_sitk(pred_img, Label)
        result_metrics['Dice - ' + cfg.organ.value] = orig_dice
        result_metrics['Volume Similarity - ' + cfg.organ.value] = orig_vs
        result_metrics['False Negative - ' + cfg.organ.value] = orig_fn
        result_metrics['False Positive - ' + cfg.organ.value] = orig_fp
        print('  Original Overlap Meassures ', orig_dice, orig_vs, orig_fn, orig_fp)

        try:
            orig_hdd = Metric.hausdorff_metric_sitk(pred_img, Label)
        except RuntimeError as err:
            print('Surface evaluation failed! Using infinity: ', err)
            orig_hdd = math.inf
        result_metrics['Hausdorff - ' + cfg.organ.value] = orig_hdd
        print('  Original Hausdorff Distance: ', orig_hdd)

        try:
            orig_mnssd, orig_mdssd, orig_stdssd, orig_maxssd = Metric.symmetric_surface_measures_sitk(pred_img, Label)
        except RuntimeError as err:
            print('Surface evaluation failed! Using infinity: ', err)
            orig_mnssd = math.inf
            orig_mdssd = math.inf
            orig_stdssd = math.inf
            orig_maxssd = math.inf

        result_metrics['Mean Symmetric Surface Distance - ' + cfg.organ.value] = orig_mnssd
        result_metrics['Median Symmetric Surface Distance - ' + cfg.organ.value] = orig_mdssd
        result_metrics['STD Symmetric Surface Distance - ' + cfg.organ.value] = orig_stdssd
        result_metrics['Max Symmetric Surface Distance - ' + cfg.organ.value] = orig_maxssd
        print('  Original Symmetric Surface Meassures ', orig_mnssd, orig_mdssd, orig_stdssd, orig_maxssd)
    else:
        LiverLabel = LiverSegLoader.get_organ_mask(Label, cfg.ORGANS.LIVER)
        LesionLabel = LiverSegLoader.get_organ_mask(Label, cfg.ORGANS.TUMORS)

        # This is necessary as in some data sets this is incorrect.
        LiverLabel.SetDirection(data_info['orig_direction'])
        LiverLabel.SetOrigin(data_info['orig_origin'])
        LiverLabel.SetSpacing(data_info['orig_spacing'])
        LesionLabel.SetDirection(data_info['orig_direction'])
        LesionLabel.SetOrigin(data_info['orig_origin'])
        LesionLabel.SetSpacing(data_info['orig_spacing'])

        result_metrics['Slices'] = data_info['orig_size'][2]
        result_metrics['Liver Volume (L)'] = Metric.get_ml_sitk(LiverLabel)
        result_metrics['Liver Volume (P)'] = Metric.get_ml_sitk(LiverSegLoader.get_organ_mask(pred_img, cfg.ORGANS.LIVER))
        result_metrics['Lesion Volume (L)'] = Metric.get_ml_sitk(LesionLabel)
        result_metrics['Lesion Volume (P)'] = Metric.get_ml_sitk(LiverSegLoader.get_organ_mask(pred_img, cfg.ORGANS.TUMORS))
        print('  Contains', result_metrics['Liver Volume (L)'], 'ml Liver Volume and',
              result_metrics['Lesion Volume (L)'], 'ml Lesion Volume.')

        orig_dice, orig_vs, orig_fn, orig_fp = Metric.overlap_measures_sitk(pred_img, LiverLabel)
        result_metrics['Dice - Liver'] = orig_dice
        result_metrics['Volume Similarity - Liver'] = orig_vs
        result_metrics['False Negative - Liver'] = orig_fn
        result_metrics['False Positive - Liver'] = orig_fp
        print('  Liver Overlap Meassures ', orig_dice, orig_vs, orig_fn, orig_fp)

        orig_dice, orig_vs, orig_fn, orig_fp = Metric.overlap_measures_sitk(pred_img, LesionLabel)
        result_metrics['Dice - Lesion'] = orig_dice
        result_metrics['Volume Similarity - Lesion'] = orig_vs
        result_metrics['False Negative - Lesion'] = orig_fn
        result_metrics['False Positive - Lesion'] = orig_fp
        print('  Lesion Overlap Meassures ', orig_dice, orig_vs, orig_fn, orig_fp)

        orig_hdd = Metric.hausdorff_metric_sitk(pred_img, LiverLabel)
        result_metrics['Hausdorff - Liver'] = orig_hdd
        print('  Liver Hausdorff Distance: ', orig_hdd)

        orig_mnsd, orig_mdsd, orig_stdsd, orig_maxsd = Metric.surface_measures_sitk(pred_img, LiverLabel)
        result_metrics['Mean Surface Distance - Liver'] = orig_mnsd
        result_metrics['Median Surface Distance - Liver'] = orig_mdsd
        result_metrics['STD Surface Distance - Liver'] = orig_stdsd
        result_metrics['Max Surface Distance - Liver'] = orig_maxsd
        print('  Liver Surface Meassures ', orig_mnsd, orig_mdsd, orig_stdsd, orig_maxsd)

    return result_metrics


def process_and_write_predictions_nii(predictions, data_info, file_number, out_path, version):
    file_number = ''.join(file_number.split())
    pred_img = Image.np_array_to_itk_image(predictions, data_info, cfg.label_background_value, cfg.adapt_resolution)
    if cfg.do_connected_component_analysis:
        pred_img = Image.extract_largest_connected_component_sitk(pred_img)
    sitk.WriteImage(pred_img, os.path.join(out_path, ('prediction' + '-' + version + '-' + file_number + '.nii')))
    return pred_img

