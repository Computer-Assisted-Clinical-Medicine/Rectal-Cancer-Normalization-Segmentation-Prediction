"""Analyze the results of the experiment"""

# %% [markdown]

# # Analyze results
# ## Import and load data

# %%

# pylint: disable=too-many-lines

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from SegClassRegBasis.evaluation import calculate_classification_metrics

from plot_utils import display_dataframe
from utils import gather_all_results

experiment_dir = Path(os.environ["experiment_dir"]) / "Normalization_Experiment"

# load data
data_dir = Path(os.environ["data_dir"])

with open(experiment_dir / "dataset.yaml", encoding="utf8") as f:
    orig_dataset = yaml.load(f, Loader=yaml.Loader)

results_class, acquisition_params = gather_all_results(task="classification")
classification_tasks = [
    c.partition("_accuracy")[0] for c in results_class if "accuracy" in c
]

results_reg, _ = gather_all_results(task="regression")
regression_tasks = [c.partition("_rmse")[0] for c in results_reg if c.endswith("_rmse")]

# %%

# See which locations are finished

print("Finished Classification")
n_finished_class = pd.DataFrame(
    index=results_class.normalization.cat.categories, columns=["2D", "3D", "total"]
)
n_finished_class["2D"] = (
    results_class.query("dimensions == 2")
    .groupby("normalization")
    .train_location.unique()
    .apply(len)
)
n_finished_class["3D"] = (
    results_class.query("dimensions == 3")
    .groupby("normalization")
    .train_location.unique()
    .apply(len)
)
n_finished_class.total = n_finished_class["2D"] + n_finished_class["3D"]
display_dataframe(n_finished_class.sort_values("total", ascending=False))

# %% [markdown]

# ## Analyze the data

# The data has multiple variable components, they are:
# - Training location
# - external (if they are from the center that was trained on)
# - postprocessed (only the biggest structure was kept)
# - version (best or final, best should usually be used)
# - before therapy (the ones afterwards are harder)
# - network (the network that was used)
# - normalization (used for training and testing)

# %%


def calculate_auc(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Calculate the AUC and other statistics for one task in the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with the columns {task}_probability_{lbl},
        {task}_ground_truth and {task}_top_prediction
    task : str
        The task name

    Returns
    -------
    pd.DataFrame
        The resulting metrics with the names as columns and the individual labels as rows
    """
    prob_prefix = f"{task}_probability_"
    prob_col_names = [c for c in df if prob_prefix in c]

    # remove missing values
    ground_truth = df[f"{task}_ground_truth"]
    probabilities = df[prob_col_names].values
    not_na = np.all(np.isfinite(probabilities), axis=-1)
    # if np.sum(~not_na):
    #     print(f"Nans found in {task}")
    mask = (ground_truth != -1) & not_na

    top_pred = df[f"{task}_top_prediction"][mask]
    probabilities = probabilities[mask]
    ground_truth = ground_truth[mask]
    if pd.api.types.is_numeric_dtype(ground_truth.dtype):
        ground_truth = ground_truth.astype(int)
    labels = np.array([val.partition(prob_prefix)[-1] for val in prob_col_names])

    if np.all(~mask):
        results = {}
    else:
        results = calculate_classification_metrics(
            prediction=top_pred,
            probabilities=probabilities,
            ground_truth=ground_truth,
            labels=labels,
        )
        del results["confusion_matrix"]

    results_per_label = {lbl: {} for lbl in labels}
    for key, vals in results.items():
        if isinstance(vals, float):
            vals = [vals] * len(labels)
        for single_val, lbl in zip(vals, labels):
            results_per_label[lbl][key] = single_val

    res_df = pd.DataFrame(results, dtype=pd.Float64Dtype())
    res_df["task"] = task
    res_df.index.name = "label"
    return res_df


res_list = []
for class_task in classification_tasks:
    # calculate AUC values
    res_tsk = (
        results_class.groupby(
            ["train_location", "normalization", "name", "version", "external", "fold"]
        )
        .apply(calculate_auc, task=class_task)
        .reset_index()
    )

    res_list.append(res_tsk)
results_class_task = pd.concat(res_list)

metrics = [
    "auc_ovo",
    "accuracy",
    "precision_mean",
]
groupby = ["task", "train_location", "normalization"]
display_dataframe(results_class_task.groupby(groupby).mean()[metrics])
display_dataframe(results_class_task.groupby(groupby).median()[metrics])

# %%

for class_task in classification_tasks:
    g = sns.catplot(
        data=results_class_task.query("version == 'best'"),
        x="auc_ovo",
        y="train_location",
        hue="normalization",
        col="external",
        kind="bar",
        legend=True,
        legend_out=True,
    )
    g.fig.suptitle(
        f"{class_task} overall Performance (version = best | before_therapy = True"
        + " | postprocessed = True)"
    )
    g.fig.subplots_adjust(top=0.87)
    plt.show()
    plt.close()

    g = sns.catplot(
        data=results_class.query("version == 'best'"),
        x=f"{class_task}_accuracy",
        y="train_location",
        hue="normalization",
        col="external",
        kind="bar",
        legend=True,
        legend_out=True,
    )
    g.fig.suptitle(
        f"{class_task} overall Performance (version = best | before_therapy = True"
        + " | postprocessed = True)"
    )
    g.fig.subplots_adjust(top=0.87)
    plt.show()
    plt.close()
