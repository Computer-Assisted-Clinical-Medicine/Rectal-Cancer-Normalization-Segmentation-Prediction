import csv
import glob
import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf

import SegmentationNetworkBasis.NetworkBasis.image as Image
import SegmentationNetworkBasis.NetworkBasis.metric as Metric
from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.NetworkBasis.util import make_csv_file

#configure logger
logger = logging.getLogger(__name__)
#disable the font manager logger
logging.getLogger('matplotlib.font_manager').disabled = True

def make_csv_header():
    header_row = ['File Number', 'Slices']
    header_row += ['Volume (L)', 'Volume (P)',
                'Dice',
                'Confusion Rate',
                'Connectivity',
                'Fragmentation',
                # 'Volume Similarity',
                'False Negative',
                'False Positive',
                'Hausdorff',
                'Mean Symmetric Surface Distance',
                # 'Median Symmetric Surface Distance',
                # 'STD Symmetric Surface Distance',
                # 'Max Symmetric Surface Distance'
                ]

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

    # check types and if not equal, convert output to target
    if pred_img.GetPixelID() != label_img.GetPixelID():
        cast = sitk.CastImageFilter()
        cast.SetOutputPixelType(label_img.GetPixelID())
        pred_img = cast.Execute(pred_img)

    result_metrics['Volume (L)'] = Metric.get_ml_sitk(label_img)

    # check if all labels are background
    if np.all(sitk.GetArrayFromImage(pred_img) == 0):
        # if that is not the case, create warning and return metrics
        if not np.all(sitk.GetArrayFromImage(label_img) == 0):
            logger.warning('Only background labels found')
            # set values for results
            result_metrics['Volume (P)'] = 0
            result_metrics['Dice' ] = 0
            result_metrics['False Negative'] = 0
            result_metrics['False Positive'] = 1
            result_metrics['Confusion Rate'] = 1
            result_metrics['Connectivity'] = 0
            result_metrics['Fragmentation'] = 1
            result_metrics['Hausdorff'] = np.NAN
            result_metrics['Mean Symmetric Surface Distance'] = np.NAN
            return result_metrics

    result_metrics['Volume (P)'] = Metric.get_ml_sitk(pred_img)

    orig_dice, orig_vs, orig_fn, orig_fp = Metric.overlap_measures_sitk(pred_img, label_img)
    result_metrics['Dice' ] = orig_dice
    # result_metrics['Volume Similarity'] = orig_vs/
    result_metrics['False Negative'] = orig_fn
    result_metrics['False Positive'] = orig_fp
    logger.info('  Original Overlap Measures: %s %s %s %s', orig_dice, orig_vs, orig_fn, orig_fp)

    cr = Metric.confusion_rate_sitk(pred_img, label_img, 1, 0)
    result_metrics['Confusion Rate'] = cr
    logger.info('  Confusion Rate: %s', cr)

    connect = Metric.get_connectivity_sitk(pred_img)
    result_metrics['Connectivity'] = connect
    logger.info('  Connectivity: %s', connect)

    frag = Metric.get_fragmentation_sitk(pred_img)
    result_metrics['Fragmentation'] = frag
    logger.info('  Fragmentation: %s', frag)

    try:
        orig_hdd = Metric.hausdorff_metric_sitk(pred_img, label_img)
    except RuntimeError as err:
        logger.error('Surface evaluation failed! Using infinity: %s', err)
        orig_hdd = np.NAN
    result_metrics['Hausdorff'] = orig_hdd
    logger.info('  Original Hausdorff Distance: %s', orig_hdd)

    try:
        orig_mnssd, orig_mdssd, orig_stdssd, orig_maxssd = Metric.symmetric_surface_measures_sitk(pred_img, label_img)
    except RuntimeError as err:
        logger.error('Surface evaluation failed! Using infinity: %s', err)
        orig_mnssd = np.NAN
        orig_mdssd = np.NAN
        orig_stdssd = np.NAN
        orig_maxssd = np.NAN

    result_metrics['Mean Symmetric Surface Distance'] = orig_mnssd
    # result_metrics['Median Symmetric Surface Distance'] = orig_mdssd
    # result_metrics['STD Symmetric Surface Distance'] = orig_stdssd
    # result_metrics['Max Symmetric Surface Distance'] = orig_maxssd
    logger.info('  Original Symmetric Surface Distance: %s (mean) %s (median) %s (STD) %s (max)', orig_mnssd, orig_mdssd, orig_stdssd, orig_maxssd)

    return result_metrics


def combine_evaluation_results_from_folds(experiment_path, eval_files):
    if len(eval_files) == 0:
        logger.info('Eval files empty, nothing to combine')
        return

    path, experiment = os.path.split(experiment_path)
    eval_mean_file_path = os.path.join(experiment_path, 'evaluation-mean-' + experiment + '.csv')
    eval_std_file_path = os.path.join(experiment_path, 'evaluation-std-' + experiment + '.csv')
    header_row = make_csv_header()
    make_csv_file(eval_mean_file_path, header_row)
    make_csv_file(eval_std_file_path, header_row)
    mean_statistics = []
    std_statistics = []

    for indiv_eval_file_path in eval_files:
        _gather_individual_results(indiv_eval_file_path, header_row,
                                    mean_statistics, std_statistics)

    for row in mean_statistics:
        with open(eval_mean_file_path, 'a', newline='') as evaluation_file:
            eval_csv_writer = csv.writer(evaluation_file, delimiter=',', quotechar='"',
                                         quoting=csv.QUOTE_MINIMAL)
            eval_csv_writer.writerow(row)

    for row in std_statistics:
        with open(eval_std_file_path, 'a', newline='') as evaluation_file:
            eval_csv_writer = csv.writer(evaluation_file, delimiter=',', quotechar='"',
                                         quoting=csv.QUOTE_MINIMAL)
            eval_csv_writer.writerow(row)


def _gather_individual_results(indiv_eval_file_path, header_row, mean_statistics, std_statistics):

    try:
        results = pd.read_csv(indiv_eval_file_path, dtype=object)
        #set index, the rest are numbers
        results = results.set_index('File Number').astype(float)
    except:
        logger.error('Could not find %s', indiv_eval_file_path)
        raise FileNotFoundError('Could not find %s', indiv_eval_file_path)

    if results.size > 0:

        average_results = results.mean().values.tolist()
        average_results = [indiv_eval_file_path.stem] + average_results
        mean_statistics.append(average_results)

        std_results = results.std().values.tolist()
        std_results = [indiv_eval_file_path.stem] + std_results
        std_statistics.append(std_results)


def make_boxplot_graphic(experiment_path, eval_files):
    if not os.path.exists(os.path.join(experiment_path, 'plots')):
        os.makedirs(os.path.join(experiment_path, 'plots'))

    if len(eval_files) == 0:
        logger.info('Eval files empty, no plots are being made')
        return

    linewidth = 2
    metrics = []
    metrics += ['Dice', 'Connectivity',
                'Fragmentation',
                'Mean Symmetric Surface Distance']

    for title in metrics:
        data = []
        labels = []
        path, experiment = os.path.split(experiment_path)

        for indiv_eval_file_path in eval_files:

            individual_results = pd.read_csv(indiv_eval_file_path, dtype=object,
                                                usecols=[title]).values
            data.append(np.squeeze(np.float32(individual_results)))

            labels.append(indiv_eval_file_path.parent.name)

        f = plt.figure(figsize=(2 * len(data) + 5, 10))
        ax = plt.subplot(111)
        [i.set_linewidth(1) for i in ax.spines.values()]

        ax.set_title(f'{experiment_path.name} {title}', pad=20)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
        '''if any(x in title for x in ['Dice', 'Fragmentation', 'Connectivity']):
            ax.set_ylim([0, 1])
        else:
            ax.set_ylabel('mm')
            if 'Hausdorff' in title:
                ax.set_ylim([0, 140])
            elif 'Mean' in title:
                ax.set_ylim([0, 40])'''

        p = plt.boxplot(data, notch=False, showmeans=True, showfliers=True, vert=True, widths=0.9,
                        patch_artist=True, labels=labels)

        # with warnings.catch_warnings():
        #     try:
        #         tight_layout()
        #     except Exception:
        #         print(title, y_label, file_path)

        plt.savefig(os.path.join(experiment_path, 'plots', title.replace(' ', '') + '.png'), transparent=True)