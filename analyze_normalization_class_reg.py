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
from tqdm import tqdm

from plot_utils import display_dataframe
from SegClassRegBasis.evaluation import calculate_classification_metrics
from utils import calculate_auc, gather_all_results

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
    index=results_class.normalization.cat.categories, columns=["2D", "total"]
)
n_finished_class["2D"] = (
    results_class.groupby("normalization").train_location.unique().apply(len)
)
n_finished_class.total = n_finished_class["2D"]
display_dataframe(n_finished_class.sort_values("total", ascending=False))

n_finished_class = pd.DataFrame(
    index=results_class.train_location.cat.categories, columns=["2D", "total"]
)
n_finished_class["2D"] = (
    results_class.groupby("train_location").normalization.unique().apply(len)
)
n_finished_class.total = n_finished_class["2D"]
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

res_list = []
for class_task in classification_tasks:
    # calculate AUC values
    res_tsk = (
        results_class.groupby(
            [
                "train_location",
                "normalization",
                "name",
                "version",
                "external",
                "fold",
            ]
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

# %%

print("Classification results for using HM-Quantile and a 2D ResNet")
res_class_all_hm_quant = (
    results_class_task.query(
        "not external & version == 'best' & normalization == 'HM_QUANTILE'"
    )
    .groupby("task")
    .mean()[
        ["auc_ovo", "auc_ovr", "accuracy", "precision", "recall", "top_2_accuracy", "std"]
    ]
    .sort_values("auc_ovo", ascending=False)
)


display_dataframe(res_class_all_hm_quant.round(2))
good_classification_tasks = list(
    res_class_all_hm_quant[res_class_all_hm_quant.auc_ovo > 0.55].index
)

print("Classification results for internal examples")
res_class_no_ext = (
    results_class_task.query("not external & version == 'best'")
    .groupby(["task", "normalization"])
    .mean()[
        ["auc_ovo", "auc_ovr", "accuracy", "precision", "recall", "top_2_accuracy", "std"]
    ]
)
display_dataframe(res_class_no_ext.loc[good_classification_tasks].round(2))

print("Classification results for external examples")
res_class_ext = (
    results_class_task.query("external & version == 'best'")
    .groupby(["task", "normalization"])
    .mean()[
        ["auc_ovo", "auc_ovr", "accuracy", "precision", "recall", "top_2_accuracy", "std"]
    ]
)
display_dataframe(res_class_ext.loc[good_classification_tasks].round(2))

print("Average AUC_OvO for all good tasks for external data")
results_class_task_good = results_class_task[
    results_class_task.task.apply(lambda t: t in good_classification_tasks)
].query("version == 'best' & external")
display_dataframe(
    results_class_task_good.groupby(["normalization", "train_location", "task"])
    .auc_ovo.mean()
    .reset_index()
    .groupby("normalization")
    .mean()
    .sort_values("auc_ovo", ascending=False)
    .round(2)
)

# %%

for class_task in good_classification_tasks:
    res_tsk = results_class_task.query(f"task == '{class_task}'")
    g = sns.catplot(
        data=res_tsk.query("version == 'best'"),
        x="auc_ovo",
        y="train_location",
        hue="normalization",
        col="external",
        kind="bar",
        legend=True,
        legend_out=True,
    )
    g.fig.suptitle(f"{class_task} overall AuC (version = best)")
    g.fig.subplots_adjust(top=0.87)
    for ax in g.axes.flat:
        ax.set_xlim(0, 1)
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
    g.fig.suptitle(f"{class_task} overall Accuracy (version = best)")
    g.fig.subplots_adjust(top=0.87)
    for ax in g.axes.flat:
        ax.set_xlim(0, 1)
    plt.show()
    plt.close()

    g = sns.catplot(
        data=results_class.query("version == 'best'"),
        x=f"{class_task}_std",
        y="train_location",
        hue="normalization",
        col="external",
        kind="box",
        legend=True,
        legend_out=True,
    )
    g.fig.suptitle(f"{class_task} overall Std (version = best)")
    g.fig.subplots_adjust(top=0.87)
    plt.show()
    plt.close()

# %%

for reg_task in regression_tasks:
    g = sns.catplot(
        data=results_reg.query("version == 'best'"),
        x=f"{reg_task}_rmse",
        y="train_location",
        hue="normalization",
        col="external",
        kind="box",
        legend=True,
        legend_out=True,
    )
    g.fig.suptitle(f"{class_task} overall RMSE (version = best)")
    g.fig.subplots_adjust(top=0.87)
    plt.show()
    plt.close()

    g = sns.catplot(
        data=results_reg.query("version == 'best'"),
        x=f"{reg_task}_std",
        y="train_location",
        hue="normalization",
        col="external",
        kind="box",
        legend=True,
        legend_out=True,
    )
    g.fig.suptitle(f"{class_task} overall STD (version = best)")
    g.fig.subplots_adjust(top=0.87)
    plt.show()
    plt.close()

# %%


def read_auc_from_files(exp_dir, res_cls, cls_tsks):
    """Calculate the metrics from the individual predictions per slice"""

    levels = {
        "train_location": res_cls.train_location.cat.categories,
        "normalization": res_cls.normalization.unique(),
        "fold": list(range(5)),
        "external": [True, False],
        "version": ["best", "final"],
        "classification_task": cls_tsks,
    }

    res = pd.DataFrame(
        index=pd.MultiIndex.from_product(levels.values(), names=levels.keys())
    )

    last_idx = (None,) * len(levels)
    for idx, _ in tqdm(res.iterrows(), total=res.shape[0]):
        (location, norm_name, fold, external, version, tsk) = idx

        exp_name = f"ResNet2D-ResNet50-{norm_name}-obj_000%-100"

        fold_dir = exp_dir / f"Normalization_{location}" / exp_name / f"fold-{fold}"
        if external:
            dataset_type = "external_testset"
            apply_dir = "apply_external_testset"
        else:
            dataset_type = "test"
            apply_dir = "apply"
        eval_file = (
            fold_dir / f"evaluation-fold-{fold}-{version}_{dataset_type}-classification.h5"
        )
        if not eval_file.exists():
            continue

        if last_idx is not None:
            same_data = np.all([i == j for i, j in zip(idx[:-1], last_idx[:-1])])
        else:
            same_data = False

        if not same_data:
            eval_file_data = pd.read_hdf(eval_file)

            res_files = []
            for f_num, _ in eval_file_data.iterrows():
                res_file = fold_dir / apply_dir / f"prediction-{f_num}-{version}.npz"
                res_dict_file = dict(np.load(res_file))
                for tsk in cls_tsks:
                    if tsk not in res_dict_file:
                        continue
                    n_elements = res_dict_file[tsk].shape[0]
                    gt_name = tsk + "_ground_truth"
                    if not gt_name in eval_file_data:
                        continue
                    g_t = eval_file_data.loc[f_num, gt_name]
                    res_dict_file[tsk + "_ground_truth"] = np.array([g_t] * n_elements)
                res_files.append(res_dict_file)

            res_dict = {
                key: np.concatenate([r[key] for r in res_files])
                for key in res_files[0].keys()
            }

        prob_prefix = f"{tsk}_probability_"
        prob_col_names = [c for c in eval_file_data if prob_prefix in c]
        labels = np.array([val.partition(prob_prefix)[-1] for val in prob_col_names])
        gt_name = f"{tsk}_ground_truth"
        if not gt_name in res_dict:
            continue
        ground_truth = res_dict[gt_name]
        probabilities = res_dict[tsk].astype(np.float32)

        if pd.api.types.is_numeric_dtype(ground_truth.dtype):
            ground_truth = ground_truth.astype(int)

        # remove missing values
        not_na = np.all(np.isfinite(probabilities), axis=-1)
        gt_missing = [g in labels for g in ground_truth.astype(str)]
        mask = gt_missing & not_na
        probabilities = probabilities[mask]
        ground_truth = ground_truth[mask]
        top_pred = labels[probabilities.argmax(axis=1)]

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
            del results["precision"]
            del results["recall"]

        for col in results:
            if col not in res:
                res[col] = pd.NA
        res.loc[idx] = results

        last_idx = idx
    return res.astype(float)


auc_file = experiment_dir / "pub" / "auc_single_files.h5"
if auc_file.exists():
    results_single_files = pd.read_hdf(auc_file).set_index(
        [
            "train_location",
            "normalization",
            "fold",
            "external",
            "version",
            "classification_task",
        ]
    )
else:
    results_single_files = read_auc_from_files(
        experiment_dir, results_class, classification_tasks
    )
    results_single_files.reset_index().to_hdf(auc_file, key="results", format="table")

# %%

for class_task in good_classification_tasks:
    res_tsk = results_single_files.reset_index().query(
        f"classification_task == '{class_task}'"
    )
    g = sns.catplot(
        data=res_tsk.query("version == 'best'"),
        x="auc_ovo",
        y="train_location",
        hue="normalization",
        col="external",
        kind="bar",
        legend=True,
        legend_out=True,
    )
    g.fig.suptitle(f"{class_task} overall AuC (version = best)")
    g.fig.subplots_adjust(top=0.87)
    for ax in g.axes.flat:
        ax.set_xlim(0, 1)
    plt.show()
    plt.close()
