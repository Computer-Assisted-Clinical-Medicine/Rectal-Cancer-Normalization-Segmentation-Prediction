import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

import SegmentationNetworkBasis.NetworkBasis.image as Image
import SegmentationNetworkBasis.NetworkBasis.metric as Metric

#configure logger
logger = logging.getLogger(__name__)
#disable the font manager logger
logging.getLogger('matplotlib.font_manager').disabled = True


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
    all_statistics_path = os.path.join(experiment_path, 'evaluation-all-files.csv')

    statistics = []
    for e in eval_files:
        data = pd.read_csv(e, sep=';')
        data['fold'] = e.parent.name
        statistics.append(data)

    # concatenate to one array
    statistics = pd.concat(statistics).sort_values('File Number')
    # write to file
    statistics.to_csv(all_statistics_path, sep=';')

    mean_statistics = statistics.groupby('fold').mean()
    mean_statistics.to_csv(eval_mean_file_path, sep=';')

    std_statistics = statistics.groupby('fold').std()
    std_statistics.to_csv(eval_std_file_path, sep=';')


def make_boxplot_graphic(experiment_path, result_file):
    if not os.path.exists(os.path.join(experiment_path, 'plots')):
        os.makedirs(os.path.join(experiment_path, 'plots'))

    if not os.path.exists(result_file):
        raise FileNotFoundError('Result file not found')

    results = pd.read_csv(result_file, sep=';')

    if results.size == 0:
        logger.info('Eval files empty, no plots are being made')
        return

    metrics = ['Dice', 'Connectivity', 'Fragmentation', 'Mean Symmetric Surface Distance']

    for m in metrics:

        groups = results.groupby('fold')
        labels = list(groups.groups.keys())
        data = groups[m].apply(list).values

        f = plt.figure(figsize=(2 * len(data) + 5, 10))
        ax = plt.subplot(111)
        [i.set_linewidth(1) for i in ax.spines.values()]

        ax.set_title(f'{experiment_path.name} {m}', pad=20)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)

        p = plt.boxplot(data, notch=False, showmeans=True, showfliers=True, vert=True, widths=0.9,
                        patch_artist=True, labels=labels)

        plt.savefig(os.path.join(experiment_path, 'plots', m.replace(' ', '') + '.png'), transparent=False)
        plt.close()