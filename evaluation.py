import csv
import glob
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


def make_csv_header():
    header_row = ['File Number', 'Slices']
    for organ in ['Vein', 'Artery']:
        header_row += [organ + ' Volume (L)', organ + ' Volume (P)',
                   'Dice - ' + organ,
                   'Confusion Rate - ' + organ,
                   'Connectivity - ' + organ,
                   'Fragmentation - ' + organ,
                   # 'Volume Similarity - ' + organ,
                   'False Negative - ' + organ,
                   'False Positive - ' + organ,
                   'Hausdorff - ' + organ,
                   'Mean Symmetric Surface Distance - ' + organ,
                   # 'Median Symmetric Surface Distance - ' + organ,
                   # 'STD Symmetric Surface Distance - ' + organ,
                   # 'Max Symmetric Surface Distance - ' + organ
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

    for organ, value in zip(['Artery', 'Vein'], [1, 2, 0]):

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
        # result_metrics['Volume Similarity - ' + organ] = orig_vs/
        result_metrics['False Negative - ' + organ] = orig_fn
        result_metrics['False Positive - ' + organ] = orig_fp
        print('  Original Overlap Measures ' + organ + ':', orig_dice, orig_vs, orig_fn, orig_fp)

        if organ == 'Artery':
            non_target_class = 2
        else:
            non_target_class = 1
        cr = Metric.confusion_rate_sitk(pred_img, label_img, value, non_target_class)
        result_metrics['Confusion Rate - ' + organ] = cr
        print('  Confusion Rate ' + organ + ':', cr)

        connect = Metric.get_connectivity_sitk(selected_pred_img)
        result_metrics['Connectivity - ' + organ] = connect
        print('  Connectivity ' + organ + ':', connect)

        frag = Metric.get_fragmentation_sitk(selected_pred_img)
        result_metrics['Fragmentation - ' + organ] = frag
        print('  Fragmentation ' + organ + ':', frag)

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
        # result_metrics['Median Symmetric Surface Distance - ' + organ] = orig_mdssd
        # result_metrics['STD Symmetric Surface Distance - ' + organ] = orig_stdssd
        # result_metrics['Max Symmetric Surface Distance - ' + organ] = orig_maxssd
        print('  Original Symmetric Surface Meassures ' + organ + ':', orig_mnssd, orig_mdssd, orig_stdssd, orig_maxssd)

    return result_metrics


def combine_evaluation_results_from_folds(experiment_path, losses, dimensions_and_architectures, evaluate_on_finetuned=False):
    path, experiment = os.path.split(experiment_path)
    eval_mean_file_path = os.path.join(experiment_path, 'evaluation-mean-' + experiment + '.csv')
    eval_std_file_path = os.path.join(experiment_path, 'evaluation-std-' + experiment + '.csv')
    header_row = make_csv_header()
    make_csv_file(eval_mean_file_path, header_row)
    make_csv_file(eval_std_file_path, header_row)
    mean_statistics = []
    std_statistics = []

    if evaluate_on_finetuned:
        for d, a in dimensions_and_architectures:
            for l1 in losses:
                for l2 in losses:
                    for do in ['DO']:  # , 'nDO']:
                        for bn in ['nBN']:  # 'BN', ]:
                            combination_name = "-".join([a.get_name() + str(d) + 'D', l1, a.get_name() + str(d) + 'D', l2, do, bn])
                            indiv_eval_file_path = os.path.join(experiment_path,
                                                    "-".join(['evaluation', combination_name]) + '.csv')
                            search_path = os.path.join(experiment_path, combination_name + '*_test', '*.csv')
                            _gather_individual_results(combination_name, search_path, indiv_eval_file_path,
                                                       header_row, mean_statistics, std_statistics)
    else:
        for d, a in dimensions_and_architectures:
            for l in losses:
                for do in ['DO']:  # , 'nDO']:
                    for bn in ['nBN']:  # 'BN', ]:
                        combination_name = "-".join([a.get_name() + str(d) + 'D', l, do, bn])
                        indiv_eval_file_path = os.path.join(experiment_path,
                                                "-".join(['evaluation', combination_name]) + '.csv')
                        search_path = os.path.join(experiment_path, combination_name + '*_test', '*.csv')
                        _gather_individual_results(combination_name, search_path, indiv_eval_file_path, header_row,
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


def _gather_individual_results(combination_name, search_path, indiv_eval_file_path, header_row, mean_statistics, std_statistics):
    results = []

    for rf in glob.glob(search_path):
        try:
            print(rf)
            individual_results = pd.read_csv(rf, dtype=object).as_matrix()
            results.append(np.float32(individual_results))
        except:
            print('Could not find', search_path)

    if len(results) > 0:
        print('Writing', indiv_eval_file_path)
        make_csv_file(indiv_eval_file_path, header_row)
        results = np.concatenate(results)
        for row in results:
            with open(indiv_eval_file_path, 'a', newline='') as evaluation_file:
                eval_csv_writer = csv.writer(evaluation_file, delimiter=',', quotechar='"',
                                             quoting=csv.QUOTE_MINIMAL)
                eval_csv_writer.writerow(row)

        average_results = np.mean(results, axis=0).tolist()
        average_results[0] = combination_name
        mean_statistics.append(average_results)

        std_results = np.std(results, axis=0).tolist()
        std_results[0] = combination_name
        std_statistics.append(std_results)


def make_boxplot_graphic(experiment_path, dimensions_and_architectures, losses, evaluate_on_finetuned=False):
    if not os.path.exists(os.path.join(experiment_path, 'plots')):
        os.makedirs(os.path.join(experiment_path, 'plots'))

    linewidth = 2
    metrics = []
    for organ in ['Vein', 'Artery', 'Combined']:
        metrics += ['Dice - ' + organ, 'Connectivity - ' + organ,
                   'Fragmentation - ' + organ,
                   'Mean Symmetric Surface Distance - ' + organ]

    for title in metrics:
        data = []
        labels = []
        path, experiment = os.path.split(experiment_path)
        if evaluate_on_finetuned:
            for d, a in dimensions_and_architectures:
                for l1 in losses:
                    for l2 in losses:
                            for do in ['DO']:  # , 'nDO']:
                                for bn in ['nBN']:  # 'BN', ]:
                                    combination_name = "-".join(
                                        [a.get_name() + str(d) + 'D', l1, a.get_name() + str(d) + 'D', l2, do, bn])
                                    indiv_eval_file_path = os.path.join(experiment_path, "-".join(
                                        ['evaluation', combination_name]) + '.csv')
                                    individual_results = pd.read_csv(indiv_eval_file_path, dtype=object,
                                                                     usecols=[title]).as_matrix()
                                    data.append(np.squeeze(np.float32(individual_results)))
                                    labels.append(combination_name)
        else:
            for d, a in dimensions_and_architectures:
                for l in losses:
                    for do in ['DO']:  # , 'nDO']:
                        for bn in ['nBN']:  # 'BN', ]:
                            combination_name = "-".join([a.get_name() + str(d) + 'D', l, do, bn])
                            indiv_eval_file_path = os.path.join(experiment_path, "-".join(
                                ['evaluation', combination_name]) + '.csv')
                            individual_results = pd.read_csv(indiv_eval_file_path, dtype=object,
                                                             usecols=[title]).as_matrix()
                            data.append(np.squeeze(np.float32(individual_results)))

                            new_name = '-'.join([a.get_name().split('N')[0], 'Net'])
                            labels.append(str(d)+ 'D' + ' ' + new_name)

        # for d, a in dimensions_and_architectures:
        #     for l in losses:
        #         for do in ['DO']:  # , 'nDO']:
        #             for bn in ['nBN']:  # 'BN', ]:
        #                 indiv_eval_file_path = os.path.join(experiment_path, "-".join(
        #                     ['evaluation', a.get_name() + str(d) + 'D', l, do, bn, experiment]) + '.csv')
        #                 individual_results = pd.read_csv(indiv_eval_file_path, dtype=object, usecols=[title]).as_matrix()
        #                 data.append(np.squeeze(np.float32(individual_results)))
        #                 labels.append("-".join([a.get_name() + str(d) + 'D', l]))

        # data.append([])
        # labels.append('')

        f = plt.figure(figsize=(2 * len(data) + 5, 10))
        ax = plt.subplot(111)
        [i.set_linewidth(1) for i in ax.spines.values()]

        # ax.set_title(title, pad=20)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
        if any(x in title for x in ['Dice', 'Fragmentation', 'Connectivity']):
            ax.set_ylim([0, 1])
        else:
            ax.set_ylabel('mm')
            if 'Hausdorff' in title:
                ax.set_ylim([0, 140])
            elif 'Mean' in title:
                ax.set_ylim([0, 40])

        p = plt.boxplot(data, notch=False, whis=[0, 100], showmeans=True, showfliers=True, vert=True, widths=0.9,
                        patch_artist=True, labels=labels)

        if "boxes" in p:
            [box.set_color(_get_color("")) for label, box in zip(labels, p["boxes"])]
            [box.set_facecolor(_get_color(label)) for label, box in zip(labels, p["boxes"])]
            [box.set_linewidth(linewidth) for box in p["boxes"]]
        if "whiskers" in p:
            for label, whisker in zip(np.repeat(labels, 2), p["whiskers"]):
                if str(2) in label:
                    whisker.set_linestyle('dashed')
                else:
                    whisker.set_linestyle('dotted')
                whisker.set_color(_get_color(label))
                whisker.set_linewidth(linewidth)
        if "medians" in p:
            for label, median in zip(labels, p["medians"]):
                # if str(dimensions[0]) in label:
                #     median.set_linestyle('dashed')
                # else:
                #     median.set_linestyle('dotted')
                median.set_color(_get_color(""))
                median.set_linewidth(linewidth)
        if "means" in p:
            for label, mean in zip(labels, p["means"]):
                if str(2) in label:
                    mean.set_marker('x')
                else:
                    mean.set_marker('+')
                mean.set_markeredgecolor(_get_color(""))
                mean.set_markerfacecolor(_get_color(""))
                mean.set_linewidth(linewidth)
        if "caps" in p:
            [cap.set_color(_get_color(label)) for label, cap in zip(np.repeat(labels, 2), p["caps"])]
            [cap.set_linewidth(linewidth) for cap in p["caps"]]
        if "fliers" in p:
            [flier.set_color(_get_color(label)) for label, flier in zip(labels, p["fliers"])]
            [flier.set_markeredgecolor(_get_color(label)) for label, flier in zip(labels, p["fliers"])]
            [flier.set_markerfacecolor(_get_color(label)) for label, flier in zip(labels, p["fliers"])]

            [flier.set_fillstyle("full") for label, flier in zip(labels, p["fliers"])]

        # with warnings.catch_warnings():
        #     try:
        #         tight_layout()
        #     except Exception:
        #         print(title, y_label, file_path)

        plt.savefig(os.path.join(experiment_path, 'plots', title.replace(' ', '') + '.png'), transparent=True)


def _get_color(label):
    colors = {"2D U-Net": "#fd192b", "2D V-Net": "#f15d5b", "2D DV-Net": "#d75fd7",
              "3D U-Net": "#fd192b", "3D V-Net": "#f15d5b", "3D DV-Net": "#d75fd7"}

    if label in colors:
        return colors[label]
    else:
        return "#000000"
