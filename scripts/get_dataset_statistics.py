import pandas as pd
import os.path
import numpy as np
import csv
import os
import SimpleITK as sitk
import SegmentationNetworkBasis.NetworkBasis.metric as Metric
import SegmentationNetworkBasis.NetworkBasis.image as Image
from SegmentationNetworkBasis import config as cfg

stats_directory = '../tmp/statistics'


def make_csv_file(eval_file_path):
    with open(eval_file_path, 'w', newline='') as evaluation_file:
        eval_csv_writer = csv.writer(evaluation_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header_row = make_csv_header()
        eval_csv_writer.writerow(header_row)
        return header_row


def make_csv_header():
    header_row = ['File Number', 'Slices', 'Voxels (#)', #'Volume',
                  'Artery Voxels (#)', 'Artery Voxels (%)', #'Artery Volume',
                  'Mean Artery HU', 'STD Artery HU', 'Min Artery HU', 'Max Artery HU',
                  'Vein Voxels (#)', 'Vein Voxels (%)', #'Vein Volume',
                  'Mean Vein HU', 'STD Vein HU', 'Min Vein HU', 'Max Vein HU',
                  'Background Voxels (#)', 'Background Voxels (%)',
                  'Mean Image HU', 'STD Image HU', 'Min Image HU', 'Max Image HU',
                  'X/Y Spacing', 'Z Spacing']
    return header_row


def write_metrics_to_csv(eval_file_path, header_row, result_metrics):
    with open(eval_file_path, 'a', newline='') as evaluation_file:
        eval_csv_writer = csv.writer(evaluation_file, delimiter=',', quotechar='"',
                                     quoting=csv.QUOTE_MINIMAL)
        eval_csv_writer.writerow(make_csv_row(header_row, result_metrics))


def make_csv_row(header_row, result_metrics):
    row = []
    for field in header_row:
        row.append(result_metrics[field])
    return row


def get_metrics(file):
    result_metrics = {}
    folder, file_number = os.path.split(file[0])
    result_metrics['File Number'] = file_number
    label_img = sitk.ReadImage(os.path.join(folder, (cfg.label_file_name_prefix + file_number + '.nii')))
    data_img = sitk.ReadImage(os.path.join(folder, (cfg.sampe_file_name_prefix + file_number + '.nii')))
    data_info = Image.get_data_info(label_img)
    result_metrics['Slices'] = data_info['orig_size'][2]
    result_metrics['X/Y Spacing'] = data_info['orig_spacing'][0]
    result_metrics['Z Spacing'] = data_info['orig_spacing'][2]

    image_statistics_filter = sitk.StatisticsImageFilter()
    image_statistics_filter.Execute(data_img)
    result_metrics['Voxels (#)'] = np.prod(data_img.GetSize())
    result_metrics['Mean Image HU'] = image_statistics_filter.GetMean()
    result_metrics['STD Image HU'] = image_statistics_filter.GetSigma()
    result_metrics['Max Image HU'] = image_statistics_filter.GetMaximum()
    result_metrics['Min Image HU'] = image_statistics_filter.GetMinimum()

    label_statistics_filter = sitk.LabelStatisticsImageFilter()
    label_statistics_filter.Execute(data_img, label_img)
    result_metrics['Background Voxels (#)'] = label_statistics_filter.GetCount(0)
    result_metrics['Background Voxels (%)'] = result_metrics['Background Voxels (#)'] / result_metrics['Voxels (#)']
    result_metrics['Artery Voxels (#)'] = label_statistics_filter.GetCount(1)
    result_metrics['Artery Voxels (%)'] = result_metrics['Artery Voxels (#)'] / result_metrics['Voxels (#)']
    result_metrics['Mean Artery HU'] = label_statistics_filter.GetMean(1)
    result_metrics['STD Artery HU'] = label_statistics_filter.GetSigma(1)
    result_metrics['Min Artery HU'] = label_statistics_filter.GetMinimum(1)
    result_metrics['Max Artery HU'] = label_statistics_filter.GetMaximum(1)
    result_metrics['Vein Voxels (#)'] = label_statistics_filter.GetCount(2)
    result_metrics['Vein Voxels (%)'] = result_metrics['Vein Voxels (#)'] / result_metrics['Voxels (#)']
    result_metrics['Mean Vein HU'] = label_statistics_filter.GetMean(2)
    result_metrics['STD Vein HU'] = label_statistics_filter.GetSigma(2)
    result_metrics['Min Vein HU'] = label_statistics_filter.GetMinimum(2)
    result_metrics['Max Vein HU'] = label_statistics_filter.GetMaximum(2)

    return result_metrics


def get_statistics(data_set):
    set_name = os.path.splitext(os.path.basename(data_set))[0]
    print('------------------', set_name, '------------------')
    stats_file = os.path.join(stats_directory, set_name + '_statistics.csv')
    header_row = make_csv_file(stats_file)
    all_files = pd.read_csv(data_set, dtype=object).as_matrix()

    for file in all_files:
        metrics = get_metrics(file)
        write_metrics_to_csv(stats_file, header_row, metrics)
        print('        ', file)

    print('------------------ done ------------------')
    print('  ')


if __name__ == '__main__':

    if not os.path.exists(stats_directory):
        os.makedirs(stats_directory)

    for data_set in ['../ircad.csv', '../btcv.csv']:
        get_statistics(data_set)





