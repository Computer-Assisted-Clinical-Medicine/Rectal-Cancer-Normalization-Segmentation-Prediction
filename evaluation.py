import math
import SegmentationNetworkBasis.NetworkBasis.metric as Metric
import SegmentationNetworkBasis.NetworkBasis.image as Image
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import os
from SegmentationNetworkBasis import config as cfg


def make_csv_header():
    header_row = ['File Number', 'Slices']
    for organ in ['Artery', 'Vein', 'Combined']:
        header_row += [organ + ' Volume (L)', organ + ' Volume (P)',
                   'Dice - ' + organ, 'Volume Similarity - ' + organ,
                   'False Negative - ' + organ,
                   'False Positive - ' + organ, 'Hausdorff - ' + organ,
                   'Mean Symmetric Surface Distance - ' + organ,
                   'Median Symmetric Surface Distance - ' + organ,
                   'STD Symmetric Surface Distance - ' + organ,
                   'Max Symmetric Surface Distance - ' + organ]

    return header_row


def evaluate_segmentation_prediction(result_metrics, prediction_path, label_path):
    pred_img = sitk.ReadImage(prediction_path)
    data_info = Image.get_data_info(pred_img)
    result_metrics['Slices'] = data_info['orig_size'][2]

    # load label for evaluation
    label_img = sitk.ReadImage(label_path)

    # This is necessary as in some data sets this is incorrect.
    label_img.SetDirection(data_info['orig_direction'])
    label_img.SetOrigin(data_info['orig_origin'])
    label_img.SetSpacing(data_info['orig_spacing'])

    for organ, value in zip(['Artery', 'Vein', 'Combined'], [1, 2, 0]):

        if organ == 'Combined':
            selected_label_img = label_img > value
            selected_pred_img = pred_img > value
        else:
            selected_label_img = label_img == value
            selected_pred_img = pred_img == value

        result_metrics[organ + ' Volume (L)'] = Metric.get_ml_sitk(selected_label_img)
        result_metrics[organ + ' Volume (P)'] = Metric.get_ml_sitk(selected_pred_img)

        orig_dice, orig_vs, orig_fn, orig_fp = Metric.overlap_measures_sitk(selected_pred_img, selected_label_img)
        result_metrics['Dice - ' + organ] = orig_dice
        result_metrics['Volume Similarity - ' + organ] = orig_vs
        result_metrics['False Negative - ' + organ] = orig_fn
        result_metrics['False Positive - ' + organ] = orig_fp
        print('  Original Overlap Meassures ' + organ + ':', orig_dice, orig_vs, orig_fn, orig_fp)

        try:
            orig_hdd = Metric.hausdorff_metric_sitk(selected_pred_img, selected_label_img)
        except RuntimeError as err:
            print('Surface evaluation failed! Using infinity: ', err)
            orig_hdd = math.inf
        result_metrics['Hausdorff - ' + organ] = orig_hdd
        print('  Original Hausdorff Distance ' + organ + ':', orig_hdd)

        try:
            orig_mnssd, orig_mdssd, orig_stdssd, orig_maxssd = Metric.symmetric_surface_measures_sitk(selected_pred_img, selected_label_img)
        except RuntimeError as err:
            print('Surface evaluation failed! Using infinity: ', err)
            orig_mnssd = math.inf
            orig_mdssd = math.inf
            orig_stdssd = math.inf
            orig_maxssd = math.inf

        result_metrics['Mean Symmetric Surface Distance - ' + organ] = orig_mnssd
        result_metrics['Median Symmetric Surface Distance - ' + organ] = orig_mdssd
        result_metrics['STD Symmetric Surface Distance - ' + organ] = orig_stdssd
        result_metrics['Max Symmetric Surface Distance - ' + organ] = orig_maxssd
        print('  Original Symmetric Surface Meassures ' + organ + ':', orig_mnssd, orig_mdssd, orig_stdssd, orig_maxssd)

    return result_metrics
