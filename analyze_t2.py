"""Analyze the T2 images"""
# %%
import os
from pathlib import Path

import pandas as pd
import seaborn as sns

from utils import gather_results

# %%

experiment_dir = Path(os.environ["experiment_dir"])

collected_results_class = []
for version in ["best", "final"]:
    loc_results = gather_results(
        experiment_dir / "Classify_Simple_Network",
        version=version,
        task="classification",
    )
    if loc_results is not None:
        loc_results["version"] = version
        collected_results_class.append(loc_results)
results_class = pd.concat(collected_results_class)

collected_results_reg = []
for version in ["best", "final"]:
    loc_results = gather_results(
        experiment_dir / "Classify_Simple_Network",
        version=version,
        task="regression",
    )
    if loc_results is not None:
        loc_results["version"] = version
        collected_results_reg.append(loc_results)

results_reg = pd.concat(collected_results_reg)

# %%

# rename for nicer plots
for res in [results_class, results_reg]:
    res.normalizing_method = res.normalizing_method.replace(
        {
            "NORMALIZING.MEAN_STD": "Mean STD",
            "NORMALIZING.HISTOGRAM_MATCHING": "HM",
            "NORMALIZING.HM_QUANTILE": "Hm Quant",
            "NORMALIZING.QUANTILE": "Quantile",
        }
    )

class_tasks = ["model_name", "sequence_variant", "field_strength", "location"]
for class_tsk in class_tasks:
    sns.catplot(
        data=results_class,
        kind="bar",
        y=f"{class_tsk}_accuracy",
        hue="version",
        x="n_conv",
        row="normalizing_method",
    )

    sns.catplot(
        data=results_class,
        kind="bar",
        y=f"{class_tsk}_accuracy",
        x="n_conv",
    )

    y_pred = results_class[f"{class_tsk}_top_prediction"]
    y = results_class[f"{class_tsk}_ground_truth"]

reg_tasks = [
    "slice_thickness",
    "repetition_time",
    "pixel_bandwidth",
    "flip_angle",
    "echo_time",
    "pixel_spacing",
]
for reg_tsk in reg_tasks:
    sns.catplot(
        data=results_reg,
        kind="box",
        y=f"{reg_tsk}_rmse",
        col="version",
        x="n_conv",
        row="normalizing_method",
    )

# %%
