"""Analyze the T2 images"""
# %%
import os
from pathlib import Path
from matplotlib import pyplot as plt

import pandas as pd

# import seaborn as sns

from SegmentationNetworkBasis import evaluation
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

# %%
# analyze classification

class_tasks = ["model_name", "sequence_variant", "field_strength", "location"]
for class_tsk in class_tasks:
    # sns.catplot(
    #     data=results_class,
    #     kind="bar",
    #     y=f"{class_tsk}_accuracy",
    #     hue="version",
    #     x="n_conv",
    #     row="normalizing_method",
    # )
    # plt.show()
    # plt.close()

    groups = results_class.groupby(["n_conv", "normalizing_method", "version"])

    def metric_func_class(dataframe, col):
        pred = dataframe[f"{col}_top_prediction"]
        truth = dataframe[f"{col}_ground_truth"]
        prob_cols = [c for c in dataframe.columns if c.startswith(f"{col}_probability_")]
        return pd.Series(
            evaluation.calculate_classification_metrics(
                prediction=pred,
                ground_truth=truth,
                probabilities=dataframe[prob_cols].values,
                labels=[l.partition("_probability_")[-1] for l in prob_cols],
            )
        )

    eval_metrics = groups.apply(metric_func_class, class_tsk)
    # plot it
    metrics = [
        "accuracy",
        "top_3_accuracy",
        "precision_mean",
        "recall_mean",
        "auc_ovo",
        "auc_ovr",
    ]
    metrics = [m for m in metrics if m in eval_metrics.columns]
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(eval_metrics.index.unique(level="normalizing_method")),
        sharex=True,
        sharey=True,
        figsize=(10, 2 + 2 * len(metrics)),
    )
    FIRST = True
    for met, ax_line in zip(metrics, axes):
        ax_line[0].set_ylabel(met.replace("_", " "))
        for norm, ax in zip(eval_metrics.index.unique(level="normalizing_method"), ax_line):
            if FIRST:
                ax.set_title(norm.replace("_", " "))
            for version in eval_metrics.index.unique(level="version"):
                data = eval_metrics.loc[(slice(None), norm, version), :]
                ax.plot(
                    data.index.get_level_values("n_conv"),
                    data[met],
                    label=f"{version}",
                )
        FIRST = False
    for ax in axes[-1]:
        ax.set_xlabel("Num conv")
    axes[0, -1].legend(bbox_to_anchor=(1, 1), loc="upper left")
    fig.suptitle(class_tsk.replace("_", " "))
    plt.tight_layout()
    plt.show()
    plt.close()

# %%

reg_tasks = [
    "slice_thickness",
    "repetition_time",
    "pixel_bandwidth",
    "flip_angle",
    "echo_time",
    "pixel_spacing",
]
for reg_tsk in reg_tasks:
    # sns.catplot(
    #     data=results_reg,
    #     kind="box",
    #     y=f"{reg_tsk}_rmse",
    #     col="version",
    #     x="n_conv",
    #     row="normalizing_method",
    # )
    # plt.show()
    # plt.close()

    groups = results_reg.groupby(["n_conv", "normalizing_method", "version"])

    def metric_func_reg(dataframe, col):
        pred = dataframe[f"{col}_mean_prediction"]
        truth = dataframe[f"{col}_ground_truth"]
        return pd.Series(
            evaluation.calculate_regression_metrics(
                prediction=pred,
                ground_truth=truth,
            )
        )

    eval_metrics = groups.apply(metric_func_reg, reg_tsk)
    # plot it
    metrics = ["rmse", "std", "max_absolute_error"]
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(eval_metrics.index.unique(level="normalizing_method")),
        sharex=True,
        sharey="row",
        figsize=(10, 2 + 2 * len(metrics)),
    )
    FIRST = True
    for met, ax_line in zip(metrics, axes):
        ax_line[0].set_ylabel(met.replace("_", " "))
        for norm, ax in zip(eval_metrics.index.unique(level="normalizing_method"), ax_line):
            if FIRST:
                ax.set_title(norm.replace("_", " "))
            for version in eval_metrics.index.unique(level="version"):
                data = eval_metrics.loc[(slice(None), norm, version), :]
                ax.plot(
                    data.index.get_level_values("n_conv"),
                    data[met],
                    label=f"{version}",
                )
        FIRST = False
    for ax in axes[-1]:
        ax.set_xlabel("Num conv")
    axes[0, -1].legend(bbox_to_anchor=(1, 1), loc="upper left")
    fig.suptitle(reg_tsk.replace("_", " "))
    plt.tight_layout()
    plt.show()
    plt.close()

# %%
