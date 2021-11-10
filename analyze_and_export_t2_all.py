"""Export segmented images"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from tqdm import tqdm

from experiment import Experiment

experiment_dir = Path(os.environ["experiment_dir"])
working_dir = experiment_dir / "Normalization_T2"
with open(experiment_dir / "dataset.yaml") as f:
    data_set = yaml.load(f, Loader=yaml.Loader)

hparam_file = working_dir / "hyperparameters.csv"
all_experiments = []
for e in pd.read_csv(hparam_file, sep=";")["path"]:
    param_file = working_dir.parent / Path(e) / "parameters.yaml"
    all_experiments.append(Experiment.from_file(param_file))

external_test_set = all_experiments[0].external_test_set

result_dir = working_dir / "Segmentations"
if not result_dir.exists():
    result_dir.mkdir()
for file in tqdm(external_test_set):
    image_path = data_set[file]["images"][0]
    image = sitk.ReadImage(str(image_path))

    fold_dirs = [exp.output_path / f"fold-{f}" for f in range(5) for exp in all_experiments]
    label_locs = [
        f_dir / "apply_external_testset" / f"prediction-{file}-best-postprocessed.nii.gz"
        for f_dir in fold_dirs
    ]
    existing_labels = [lbl.exists() for lbl in label_locs]
    found_labels = np.array(label_locs)[existing_labels]
    labels_list_sitk = [sitk.ReadImage(str(lbl)) for lbl in found_labels]
    labels_list_sitk_resampled = [labels_list_sitk[0]]
    for lbl in labels_list_sitk[1:]:
        labels_list_sitk_resampled.append(
            sitk.Resample(
                lbl,
                referenceImage=labels_list_sitk[0],
                interpolator=sitk.sitkNearestNeighbor,
                outputPixelType=sitk.sitkUInt8,
                useNearestNeighborExtrapolator=True,
            )
        )
    labels_np = np.array(
        [sitk.GetArrayFromImage(lbl) for lbl in labels_list_sitk_resampled]
    )
    # use the median
    labels_np = np.median(labels_np, axis=0)
    # make int
    labels_np = labels_np.round().astype(int)
    labels_sitk = sitk.GetImageFromArray(labels_np)
    labels_sitk.CopyInformation(labels_list_sitk[0])
    # resample to image
    final_labels = sitk.Resample(
        labels_sitk,
        referenceImage=image,
        interpolator=sitk.sitkNearestNeighbor,
        outputPixelType=sitk.sitkUInt8,
        useNearestNeighborExtrapolator=True,
    )
    save_path = result_dir / f"seg_{file.replace('_t', '_T2_axial_')}.nii.gz"
    sitk.WriteImage(final_labels, str(save_path))
    save_path = result_dir / f"seg_{file.replace('_t', '_T2_axial_')}_not_resampled.nii.gz"
    sitk.WriteImage(labels_sitk, str(save_path))
