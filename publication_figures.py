"""Make the figures for the paper"""

# pylint:disable=too-many-lines, invalid-name

# %%

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import SimpleITK as sitk
import yaml
from IPython import display
from matplotlib.transforms import Bbox
from tqdm import tqdm

from gan_normalization import GanDiscriminators
from plot_utils import create_axes, display_dataframe, display_markdown, plot_significance
from SegClassRegBasis import normalization
from utils import calculate_auc, gather_all_results, hatched_histplot


def turn_ticks(axes_to_turn):
    for label in axes_to_turn.get_xticklabels():
        label.set_ha("right")
        label.set_rotation(40)


def save_pub(filename, **kwargs):
    """Save the figure for publication"""
    pub_dir = Path(os.environ["experiment_dir"]) / "Normalization_Experiment" / "pub"
    if not pub_dir.exists():
        pub_dir.mkdir()
    plt.savefig(pub_dir / f"{filename}.png", dpi=600, **kwargs)
    plt.savefig(pub_dir / f"{filename}.eps", dpi=600, **kwargs)
    plt.savefig(pub_dir / f"{filename}.pgf", dpi=600, **kwargs)
    plt.savefig(pub_dir / f"{filename}.pdf", dpi=600, **kwargs)

    if matplotlib.rcParams["backend"] == "pgf":
        display.display(display.Image(filename=pub_dir / f"{filename}.png"))
    else:
        plt.show()
    plt.close()


# %%

# matplotlib.use("pgf")
sns.set_style()
sns.set_context("paper")
plt.style.use("seaborn-paper")

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

DISS_TEXTWIDTH = 394.78204 / 72

plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams["text.usetex"] = True
plt.rcParams["pgf.rcfonts"] = False

sns.set_palette(
    sns.color_palette(["#21a7bf", "#d75109", "#34c165", "#f29e02", "#9d060d", "#515b34"])
)

plt.rcParams["hatch.color"] = "#FFFFFF"

# %%

print("Load the results")

experiment_dir = Path(os.environ["experiment_dir"]) / "Normalization_Experiment"

# load data
data_dir = Path(os.environ["data_dir"])

with open(experiment_dir / "dataset.yaml", encoding="utf8") as f:
    orig_dataset = yaml.load(f, Loader=yaml.Loader)


results_seg, acquisition_params_seg = gather_all_results(task="segmentation")

results_seg = results_seg.copy()
results_seg["do_batch_normalization"] = results_seg["do_batch_normalization"].astype(bool)

results_class, acquisition_params = gather_all_results(task="classification")

# add pCR as task
results_class["pCR_accuracy"] = 0
results_class.loc[results_class.dworak_top_prediction >= 3, "pCR_top_prediction"] = "yes"
results_class.loc[results_class.dworak_top_prediction < 3, "pCR_top_prediction"] = "no"
results_class.loc[results_class.dworak_top_prediction < 0, "pCR_top_prediction"] = -1

results_class.loc[results_class.dworak_ground_truth >= 3, "pCR_ground_truth"] = "yes"
results_class.loc[results_class.dworak_ground_truth < 3, "pCR_ground_truth"] = "no"
results_class.loc[results_class.dworak_ground_truth < 0, "pCR_ground_truth"] = -1

results_class["pCR_probability_yes"] = (
    results_class.dworak_probability_3 + results_class.dworak_probability_4
)
results_class["pCR_probability_no"] = 1 - results_class.pCR_probability_yes

results_class.pCR_accuracy = (
    results_class.dworak_ground_truth == results_class.dworak_top_prediction
).astype(float)
results_class["pCR_std"] = results_class.dworak_std

results_reg, _ = gather_all_results(task="regression")
regression_tasks = [c.partition("_rmse")[0] for c in results_reg if c.endswith("_rmse")]

classification_tasks = [
    c.partition("_accuracy")[0] for c in results_class if "accuracy" in c
]

# %%

# set the experiment type
experiment_types = ["All", "Except-One", "Single-Center", "Single-Center-All"]
for df in [results_seg, results_class, results_reg]:
    df.loc[df.train_location.apply(lambda x: "Not" in x), "experiment_type"] = "Except-One"
    df.loc[
        df.train_location.apply(lambda x: "Not" not in x), "experiment_type"
    ] = "Single-Center"
    df.loc[
        df.train_location.apply(lambda x: "-all" in x), "experiment_type"
    ] = "Single-Center-All"
    df.loc[df.train_location == "all", "experiment_type"] = "All"
    df.experiment_type = pd.Categorical(df.experiment_type, categories=experiment_types)

extended_experiment_types = [
    "All",
    "Except-One-Internal",
    "Except-One-External",
    "Single-Center-Internal",
    "Single-Center-External",
    "Single-Center-All-Internal",
    "Single-Center-All-External",
]
for df in [results_seg, results_class, results_reg]:
    not_centers = df.train_location.apply(lambda x: "Not" in x)
    df.loc[
        (df.experiment_type == "Except-One") & (~df.external),
        "extended_experiment_type",
    ] = "Except-One-Internal"
    df.loc[
        (df.experiment_type == "Except-One") & df.external,
        "extended_experiment_type",
    ] = "Except-One-External"
    df.loc[
        (df.experiment_type == "Single-Center") & (~df.external),
        "extended_experiment_type",
    ] = "Single-Center-Internal"
    df.loc[
        (df.experiment_type == "Single-Center") & df.external,
        "extended_experiment_type",
    ] = "Single-Center-External"
    df.loc[df.train_location == "all", "extended_experiment_type"] = "All"
    df.loc[
        (df.experiment_type == "Single-Center-All") & (~df.external),
        "extended_experiment_type",
    ] = "Single-Center-All-Internal"
    df.loc[
        (df.experiment_type == "Single-Center-All") & df.external,
        "extended_experiment_type",
    ] = "Single-Center-All-External"
    df.extended_experiment_type = pd.Categorical(
        df.extended_experiment_type, categories=extended_experiment_types
    )

# %%

print("Drop the Single-Center-All experiment")
for df in [results_seg, results_class, results_reg]:
    to_drop = df.index[df.experiment_type == "Single-Center-All"]
    df.drop(index=to_drop, inplace=True)
    # to_drop = df.index[~ df.from_study]
    # df.drop(index=to_drop, inplace=True)
    for col in df:
        if df.dtypes[col].name == "category":
            df[col] = df[col].cat.remove_unused_categories()
experiment_types = ["All", "Except-One", "Single-Center"]
extended_experiment_types = [
    "All",
    "Except-One-Internal",
    "Except-One-External",
    "Single-Center-Internal",
    "Single-Center-External",
]


# %%

print("Calculate AUC values")

res_list = []
for class_task in classification_tasks:
    # calculate AUC values
    res_tsk = results_class.groupby(
        [
            "experiment_type",
            "extended_experiment_type",
            "train_location",
            "normalization",
            "name",
            "version",
            "external",
            "fold",
        ]
    ).apply(calculate_auc, task=class_task)
    index_names = res_tsk.index.names
    if index_names[-1] is None:
        res_tsk.index = res_tsk.index.set_names(index_names[:-1] + ["label"])

    res_list.append(res_tsk.reset_index())
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
    res_class_all_hm_quant[res_class_all_hm_quant.auc_ovo > 0.58].index
)


# %%

display_markdown("# Publication figures")
print("Defining the new names")

new_names_paper = {
    "GAN_DISCRIMINATORS": "GAN",
    "GAN_DISCRIMINATORS_3_64_0.50": "GAN_3_64",
    "GAN_DISCRIMINATORS_3_64_0.50_BetterConv": "GAN_3_64_BC",
    "GAN_DISCRIMINATORS_3_64_0.50_BetterConv_0.00001": "GAN-Def",
    "GAN_DISCRIMINATORS_3_64_0.50_BetterConv_0.00001_seg": "GAN-Seg",
    "GAN_DISCRIMINATORS_3_64_0.50_BetterConv_0.00001_all_image": "GAN-Img",
    "GAN_DISCRIMINATORS_3_64_0.50_BetterConv_0.00001_WINDOW": "GAN-Win",
    "GAN_DISCRIMINATORS_3_64_0.50_BetterConv_0.00001_WINDOW_seg": "GAN_3_64_BC_win_seg",
    "GAN_DISCRIMINATORS_3_64_n-skp_BetterConv_0.00001_WINDOW": "GAN-No-ed",
    "QUANTILE": "Perc",
    "HM_QUANTILE": "Perc-HM",
    "MEAN_STD": "M-Std",
    "HISTOGRAM_MATCHING": "HM",
    "WINDOW": "Win",
    "UNet2D": "UNet",
    "DeepLabv3plus2D": "DeepLabV3+",
    "Frankfurt": "Center 1",
    "Regensburg": "Center 2",
    "Mannheim-not-from-study": "Center 3b",
    "Mannheim": "Center 3a",
    "Wuerzburg": "Center 4",
    "Regensburg-UK": "Center 5",
    "Freiburg": "Center 6",
    "Not-Frankfurt": "Not Center 1",
    "Not-Regensburg": "Not Center 2",
    "Not-Mannheim": "Not Center 3",
}
center_order_paper = [
    "Center 1",
    "Center 2",
    "Center 3a",
    "Center 3b",
    "Center 4",
    "Center 5",
    "Center 6",
]
new_names_diss = {
    "GAN_DISCRIMINATORS": "GAN",
    "GAN_DISCRIMINATORS_3_64_0.50": "GAN_3_64",
    "GAN_DISCRIMINATORS_3_64_0.50_BetterConv": "GAN_3_64_BC",
    "GAN_DISCRIMINATORS_3_64_0.50_BetterConv_0.00001": "GAN-Def",
    "GAN_DISCRIMINATORS_3_64_0.50_BetterConv_0.00001_seg": "GAN-Seg",
    "GAN_DISCRIMINATORS_3_64_0.50_BetterConv_0.00001_all_image": "GAN-Img",
    "GAN_DISCRIMINATORS_3_64_0.50_BetterConv_0.00001_WINDOW": "GAN-Win",
    "GAN_DISCRIMINATORS_3_64_0.50_BetterConv_0.00001_WINDOW_seg": "GAN_3_64_BC_win_seg",
    "GAN_DISCRIMINATORS_3_64_n-skp_BetterConv_0.00001_WINDOW": "GAN-No-ed",
    "QUANTILE": "Perc",
    "HM_QUANTILE": "Perc-HM",
    "MEAN_STD": "M-Std",
    "HISTOGRAM_MATCHING": "HM",
    "WINDOW": "Win",
    "UNet2D": "UNet",
    "DeepLabv3plus2D": "DeepLabV3+",
    "Frankfurt": "Center 1",
    "Regensburg": "Center 11",
    "Mannheim-not-from-study": "In-house",
    "Mannheim": "Center 13",
    "Wuerzburg": "Center 12",
    "Regensburg-UK": "Center 5",
    "Freiburg": "Center 8",
    "Not-Frankfurt": "Not Center 1",
    "Not-Regensburg": "Not Center 11",
    "Not-Mannheim": "Not Center 13",
}
center_order_diss = [
    "Center 1",
    "Center 5",
    "Center 8",
    "Center 11",
    "Center 12",
    "Center 13",
    "In-house",
]
norm_order = [
    "Perc",
    "HM",
    "Perc-HM",
    "M-Std",
    "Win",
    "GAN-Def",
    "GAN-Seg",
    "GAN-Img",
    "GAN-Win",
    "GAN-No-ed",
]
external_order = ["test", False, True]

# change this to adjust the center names
# NEW_NAMES = new_names_paper
# CENTER_ORDER = center_order_paper
NEW_NAMES = new_names_diss
CENTER_ORDER = center_order_diss

results_seg_new_names = results_seg.copy()
results_class_new_names = results_class.copy()
results_class_task_new_names = results_class_task.copy()
results_reg_new_names = results_reg.copy()

for df in [
    results_seg_new_names,
    results_class_new_names,
    results_class_task_new_names,
    results_reg_new_names,
]:
    df.replace(NEW_NAMES, inplace=True)
    df.normalization = df.normalization.cat.remove_categories(
        [c for c in df.normalization.cat.categories if c not in norm_order]
    )
    df.normalization = df.normalization.cat.reorder_categories(norm_order)

# for some reason, drop does not work
results_seg_new_names = results_seg_new_names[results_seg_new_names.normalization.notna()]
results_class_new_names = results_class_new_names[
    results_class_new_names.normalization.notna()
]
results_class_task_new_names = results_class_task_new_names[
    results_class_task_new_names.normalization.notna()
]
results_reg_new_names = results_reg_new_names[results_reg_new_names.normalization.notna()]

# %%

for experiment_type in experiment_types:

    print(f"Performance on {experiment_type.replace('_', ' ')}")

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(20 / 2.54, 15 / 2.54))

    data_seg = results_seg_new_names.query(
        "before_therapy & postprocessed & name != 'combined_models' & version == 'best'"
        f" & experiment_type == '{experiment_type}'"
    )
    data_class = results_class_task_new_names.query(
        f"experiment_type == '{experiment_type}' & version == 'best'"
    )
    data_reg = results_reg_new_names.query(
        f"experiment_type == '{experiment_type}' & version == 'best'"
    )

    sns.boxplot(
        data=data_seg.query("do_batch_normalization"),
        x="normalization",
        y="Dice",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Segmentation Batch Norm")
    turn_ticks(axes[0, 0])
    axes[0, 0].set_ylim(-0.05, 1.05)

    sns.boxplot(
        data=data_seg.query("not do_batch_normalization"),
        x="normalization",
        y="Dice",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Segmentation without Batch Norm")
    turn_ticks(axes[0, 1])
    axes[0, 1].set_ylim(-0.05, 1.05)

    for ax, task in zip(axes.flat[2:], good_classification_tasks):
        sns.boxplot(
            data=data_class.query(f"task == '{task}'"),
            x="normalization",
            y="auc_ovo",
            ax=ax,
        )
        ax.set_title(task.replace("_", " ").capitalize())
        ax.set_ylabel("AUC")
        ax.set_ylim(0.2 - 0.05, 1.05)
        turn_ticks(ax)

    sns.boxplot(
        data=data_reg,
        x="normalization",
        y="age_rmse",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Age")
    axes[1, 1].set_ylabel("RMSE")
    axes[1, 1].set_ylim(-1, 25)
    turn_ticks(axes[1, 1])

    axes[1, 2].remove()

    plt.tight_layout()
    save_pub(f"{experiment_type}_summary")
    plt.show()
    plt.close()

# %%

print("Performance in all experiments")

fig, axes = plt.subplots(
    nrows=3, ncols=2, figsize=(DISS_TEXTWIDTH * 0.9, 20 / 2.54), sharey="row"
)

data_seg = results_seg_new_names.query(
    "before_therapy & postprocessed & name != 'combined_models' & version == 'best'"
).copy()
data_class = results_class_task_new_names.query("version == 'best'").copy()
data_reg = results_reg_new_names.query("version == 'best'").copy()

data_seg = data_seg[np.logical_or(data_seg.external, data_seg.experiment_type == "All")]
data_class = data_class[
    np.logical_or(data_class.external, data_class.experiment_type == "All")
]
data_reg = data_reg[np.logical_or(data_reg.external, data_reg.experiment_type == "All")]

sns.boxplot(
    data=data_seg.query("do_batch_normalization"),
    x="normalization",
    y="Dice",
    hue="experiment_type",
    hue_order=experiment_types,
    ax=axes[0, 0],
)
axes[0, 0].set_title("Segmentation with Batch Norm")
turn_ticks(axes[0, 0])
axes[0, 0].legend([], [], frameon=False)
axes[0, 0].set_ylim(-0.05, 0.95)

sns.boxplot(
    data=data_seg.query("not do_batch_normalization"),
    x="normalization",
    y="Dice",
    hue="experiment_type",
    ax=axes[0, 1],
)
axes[0, 1].set_title("Segmentation without Batch Norm")
turn_ticks(axes[0, 1])
axes[0, 1].legend([], [], frameon=False)
axes[0, 1].set_ylim(-0.05, 0.95)

for ax, task in zip(axes.flat[2:], good_classification_tasks):
    sns.boxplot(
        data=data_class.query(f"task == '{task}'"),
        x="normalization",
        y="auc_ovo",
        hue="experiment_type",
        hue_order=experiment_types,
        ax=ax,
    )
    ax.set_title(task.replace("_", " ").capitalize())
    ax.set_ylabel("AUC")
    turn_ticks(ax)
    ax.legend([], [], frameon=False)

axes[0, 1].set_ylabel("")
axes[1, 1].set_ylabel("")

for ax in axes.flat[:4]:
    ax.set_xlabel("")

sns.boxplot(
    data=data_reg,
    x="normalization",
    y="age_rmse",
    hue="experiment_type",
    hue_order=experiment_types,
    ax=axes[2, 0],
)
axes[2, 0].set_title("Age")
axes[2, 0].set_ylabel("RMSE")
turn_ticks(axes[2, 0])
axes[2, 0].set_ylim(-2, 45)
legend = axes[2, 0].legend(
    borderaxespad=0,
    title="Experiment Type",
)

axes[2, 1].remove()

plt.tight_layout()

legend.set(bbox_to_anchor=[1.1, 0, 0, 1])

save_pub("all_experiment_types_summary")
plt.show()
plt.close()
# %%

print("Performance in all experiments as bar graph")

fig, axes = plt.subplots(
    nrows=3, ncols=2, figsize=(DISS_TEXTWIDTH * 0.9, 20 / 2.54), sharey="row"
)

data_seg = results_seg_new_names.query(
    "before_therapy & postprocessed & name != 'combined_models' & version == 'best'"
).copy()
data_class = results_class_task_new_names.query("version == 'best'").copy()
data_reg = results_reg_new_names.query("version == 'best'").copy()

data_seg = data_seg[np.logical_or(data_seg.external, data_seg.experiment_type == "All")]
data_class = data_class[
    np.logical_or(data_class.external, data_class.experiment_type == "All")
]
data_reg = data_reg[np.logical_or(data_reg.external, data_reg.experiment_type == "All")]

bar_settings = dict(
    errwidth=1,
    hue="experiment_type",
    hue_order=experiment_types,
)

sns.barplot(
    data=data_seg.query("do_batch_normalization"),
    x="normalization",
    y="Dice",
    ax=axes[0, 0],
    **bar_settings,
)
axes[0, 0].set_title("Segmentation Batch Norm")
turn_ticks(axes[0, 0])
axes[0, 0].legend([], [], frameon=False)
axes[0, 0].set_ylim(0, 0.78)

sns.barplot(
    data=data_seg.query("not do_batch_normalization"),
    x="normalization",
    y="Dice",
    ax=axes[0, 1],
    **bar_settings,
)
axes[0, 1].set_title("Segmentation without Batch Norm")
turn_ticks(axes[0, 1])
axes[0, 1].legend([], [], frameon=False)
axes[0, 1].set_ylim(0, 0.78)

for ax, task in zip(axes.flat[2:], good_classification_tasks):
    sns.barplot(
        data=data_class.query(f"task == '{task}'"),
        x="normalization",
        y="auc_ovo",
        ax=ax,
        **bar_settings,
    )
    ax.set_title(task.replace("_", " ").capitalize())
    ax.set_ylabel("AUC")
    turn_ticks(ax)
    ax.legend([], [], frameon=False)
axes[1, 0].set_ylim(0.4, 1)
# axes[1, 0].set_ylim(0.45, 0.7)

legend = axes[2, 0].legend(
    bbox_to_anchor=(0, 0),
    loc="upper left",
    borderaxespad=0,
    title="Experiment Type",
)

axes[0, 1].set_ylabel("")
axes[1, 1].set_ylabel("")

for ax in axes.flat[:4]:
    ax.set_xlabel("")

sns.barplot(
    data=data_reg,
    x="normalization",
    y="age_rmse",
    ax=axes[2, 0],
    **bar_settings,
)
axes[2, 0].set_title("Age")
axes[2, 0].set_ylabel("RMSE")
turn_ticks(axes[2, 0])
axes[2, 0].set_ylim(10, 20)
legend = axes[2, 0].legend(
    borderaxespad=0,
    title="Experiment Type",
)

axes[2, 1].remove()

plt.tight_layout()

legend.set(bbox_to_anchor=[1.1, 0, 0, 1])
save_pub("all_experiment_types_summary_bars")
plt.show()
plt.close()

# %%

print("Performance in all experiments as bar graph without nBN")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5.46, 5.46))

data_seg = results_seg_new_names.query(
    "before_therapy & postprocessed & name != 'combined_models' & version == 'best'"
).copy()
data_class = results_class_task_new_names.query("version == 'best'").copy()
data_reg = results_reg_new_names.query("version == 'best'").copy()

data_seg = data_seg[np.logical_or(data_seg.external, data_seg.experiment_type == "All")]
data_class = data_class[
    np.logical_or(data_class.external, data_class.experiment_type == "All")
]
data_reg = data_reg[np.logical_or(data_reg.external, data_reg.experiment_type == "All")]

bar_settings = dict(
    errwidth=1,
    hue="experiment_type",
    hue_order=experiment_types,
)

sns.barplot(
    data=data_seg.query("do_batch_normalization"),
    x="normalization",
    y="Dice",
    ax=axes[0, 0],
    **bar_settings,
)
axes[0, 0].set_title("Segmentation")
turn_ticks(axes[0, 0])
axes[0, 0].legend([], [], frameon=False)
axes[0, 0].set_ylim(0, 0.8)

for ax, task in zip(axes.flat[1:], good_classification_tasks):
    sns.barplot(
        data=data_class.query(f"task == '{task}'"),
        x="normalization",
        y="auc_ovo",
        ax=ax,
        **bar_settings,
    )
    ax.set_title(task.replace("_", " ").capitalize())
    ax.set_ylabel("AUC")
    turn_ticks(ax)
    if task != "sex":
        ax.legend([], [], frameon=False)
axes[0, 1].set_ylim(0.2, 1)
axes[1, 0].set_ylim(0.45, 0.7)

sns.barplot(
    data=data_reg,
    x="normalization",
    y="age_rmse",
    ax=axes[1, 1],
    **bar_settings,
)
axes[1, 1].set_title("Age")
axes[1, 1].set_ylabel("RMSE")
turn_ticks(axes[1, 1])
axes[1, 1].set_ylim(10, 20)
axes[1, 1].legend([], [], frameon=False)

plt.tight_layout()
axes[0, 1].legend().set(
    bbox_to_anchor=[0.95, 0, 0, 1],
    title="Experiment Type",
)
save_pub(
    "all_experiment_types_summary_bars_nnBN",
    bbox_inches=Bbox.from_extents(0.1, 0.1, 6.5, 5.36),
)
plt.show()
plt.close()
# %%

print("Performance in all experiments as bar graph with extended experiments")

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20 / 2.54, 16 / 2.54))

data_seg = results_seg_new_names.query(
    "before_therapy & postprocessed & name != 'combined_models' & version == 'best'"
).copy()
data_class = results_class_task_new_names.query("version == 'best'").copy()
data_reg = results_reg_new_names.query("version == 'best'").copy()

bar_settings = dict(
    errwidth=0.8,
    hue="extended_experiment_type",
    hue_order=extended_experiment_types,
)

sns.barplot(
    data=data_seg.query("do_batch_normalization"),
    x="normalization",
    y="Dice",
    ax=axes[0, 0],
    **bar_settings,
)
axes[0, 0].set_title("Segmentation Batch Norm")
turn_ticks(axes[0, 0])
axes[0, 0].legend([], [], frameon=False)
axes[0, 0].set_ylim(0, 0.8)

sns.barplot(
    data=data_seg.query("not do_batch_normalization"),
    x="normalization",
    y="Dice",
    ax=axes[0, 1],
    **bar_settings,
)
axes[0, 1].set_title("Segmentation without Batch Norm")
turn_ticks(axes[0, 1])
axes[0, 1].legend([], [], frameon=False)
axes[0, 1].set_ylim(0, 0.8)

for ax, task in zip(axes.flat[2:], good_classification_tasks):
    sns.barplot(
        data=data_class.query(f"task == '{task}'"),
        x="normalization",
        y="auc_ovo",
        ax=ax,
        **bar_settings,
    )
    ax.set_title(task.replace("_", " ").capitalize())
    ax.set_ylabel("AUC")
    turn_ticks(ax)
    if task != "sex":
        ax.legend([], [], frameon=False)
axes[0, 2].set_ylim(0.2, 1)
axes[1, 0].set_ylim(0.45, 0.7)

legend = axes[0, 2].legend(
    bbox_to_anchor=(0, 0),
    loc="upper left",
    borderaxespad=0,
    title="Experiment Type",
)

sns.barplot(
    data=data_reg,
    x="normalization",
    y="age_rmse",
    **bar_settings,
    ax=axes[1, 1],
)
axes[1, 1].set_title("Age")
axes[1, 1].set_ylabel("RMSE")
turn_ticks(axes[1, 1])
axes[1, 1].set_ylim(10, 20)
axes[1, 1].legend([], [], frameon=False)

axes[1, 2].remove()

plt.tight_layout()
legend.set(bbox_to_anchor=[-0.0, -1.45, 0, 1])
save_pub("all_extended_experiment_types_summary_bars")
plt.show()
plt.close()

# %%

print("Performance in all experiments as par graph with extended experiments without nBN")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20 / 2.54, 16 / 2.54))

data_seg = results_seg_new_names.query(
    "before_therapy & postprocessed & name != 'combined_models' & version == 'best'"
).copy()
data_class = results_class_task_new_names.query("version == 'best'").copy()
data_reg = results_reg_new_names.query("version == 'best'").copy()

bar_settings = dict(
    errwidth=1,
    hue="extended_experiment_type",
    hue_order=extended_experiment_types,
)

sns.barplot(
    data=data_seg.query("do_batch_normalization"),
    x="normalization",
    y="Dice",
    ax=axes[0, 0],
    **bar_settings,
)
axes[0, 0].set_title("Segmentation")
turn_ticks(axes[0, 0])
axes[0, 0].legend([], [], frameon=False)
axes[0, 0].set_ylim(0, 0.8)

for ax, task in zip(axes.flat[1:], good_classification_tasks):
    sns.barplot(
        data=data_class.query(f"task == '{task}'"),
        x="normalization",
        y="auc_ovo",
        ax=ax,
        **bar_settings,
    )
    ax.set_title(task.replace("_", " ").capitalize())
    ax.set_ylabel("AUC")
    turn_ticks(ax)
    if task != "sex":
        ax.legend([], [], frameon=False)
axes[0, 1].set_ylim(0.2, 1)
axes[1, 0].set_ylim(0.45, 0.7)

sns.barplot(
    data=data_reg,
    x="normalization",
    y="age_rmse",
    ax=axes[1, 1],
    **bar_settings,
)
axes[1, 1].set_title("Age")
axes[1, 1].set_ylabel("RMSE")
turn_ticks(axes[1, 1])
axes[1, 1].set_ylim(10, 20)
axes[1, 1].legend([], [], frameon=False)

plt.tight_layout()
axes[0, 1].legend().set(
    bbox_to_anchor=[0.97, 0, 0, 1],
    title="Experiment Type",
)
save_pub("all_extended_experiment_types_summary_bars_nnBN")
plt.show()
plt.close()
# %%

print("Make a table")

results_seg = results_seg_new_names.query(
    "before_therapy & postprocessed & name != 'combined_models' & version == 'best'"
).copy()
data_class = results_class_task_new_names.query("version == 'best'").copy()
data_reg = results_reg_new_names.query("version == 'best'").copy()

res_table_list = []

res_table_list.append(
    results_seg.query("do_batch_normalization")
    .groupby(["normalization", "extended_experiment_type"])
    .Dice.mean()
    .unstack(level=0)
    .round(2)
)
res_table_list.append(
    results_seg.query("not do_batch_normalization")
    .groupby(["normalization", "extended_experiment_type"])
    .Dice.mean()
    .unstack(level=0)
    .round(2)
)

for ax, task in zip(axes.flat[2:], good_classification_tasks):
    res_table_list.append(
        data_class.query(f"task == '{task}'")
        .groupby(["normalization", "extended_experiment_type"])
        .auc_ovo.mean()
        .unstack(level=0)
        .round(2)
    )
    data = data_class.query(f"task == '{task}'")

res_table_list.append(
    data_reg.groupby(["normalization", "extended_experiment_type"])
    .age_rmse.mean()
    .unstack(level=0)
    .round(2)
)

res_table = pd.concat(res_table_list)
display_dataframe(res_table)

# %%

print("Make a reduced table without nBN")

results_seg = results_seg_new_names.query(
    "before_therapy & postprocessed & name != 'combined_models' & version == 'best'"
    + " & (experiment_type == 'All' or experiment_type == 'Except-One')"
).copy()
data_class = results_class_task_new_names.query(
    "version == 'best' & (experiment_type == 'All' or experiment_type == 'Except-One')"
).copy()
data_reg = results_reg_new_names.query(
    "version == 'best' & (experiment_type == 'All' or experiment_type == 'Except-One')"
).copy()


res_table = pd.DataFrame(
    index=results_seg.normalization.cat.categories,
    columns=pd.MultiIndex.from_product(
        [
            ["Segmentation", "Sex", "Dworak", "Age"],
            ["All", "Except-One-Internal", "Except-One-External"],
        ]
    ),
)

res_table["Segmentation"] = (
    results_seg.query("do_batch_normalization")
    .groupby(["normalization", "extended_experiment_type"])
    .Dice.mean()
    .dropna()
    .unstack(level=1)
    .round(2)
)

res_table["Sex"] = (
    data_class.query("task == 'sex'")
    .groupby(["normalization", "extended_experiment_type"])
    .auc_ovo.mean()
    .dropna()
    .unstack(level=1)
    .round(2)
)

res_table["Dworak"] = (
    data_class.query("task == 'dworak'")
    .groupby(["normalization", "extended_experiment_type"])
    .auc_ovo.mean()
    .dropna()
    .unstack(level=1)
    .round(2)
)

res_table["Age"] = (
    data_reg.groupby(["normalization", "extended_experiment_type"])
    .age_rmse.mean()
    .dropna()
    .unstack(level=1)
    .round(2)
)

styler = res_table.style
styler = styler.format(precision=2)
display_dataframe(styler)
print(
    styler.to_latex(
        convert_css=True,
        caption="Caption.",
        label="results_table",
    )
)

# %%

print("Make a reduced table without nBN without internal data")

results_seg = results_seg_new_names.query(
    "before_therapy & postprocessed & name != 'combined_models' & version == 'best'"
).copy()
data_class = results_class_task_new_names.query("version == 'best'").copy()
data_reg = results_reg_new_names.query("version == 'best'").copy()


res_table = pd.DataFrame(
    index=results_seg.normalization.cat.categories,
    columns=pd.MultiIndex.from_product(
        [
            ["Segmentation", "Sex", "Dworak", "Age"],
            [
                "All",
                "Except-One-Internal",
                "Except-One-External",
                "Single-Center-Internal",
                "Single-Center-External",
            ],
        ]
    ),
)

res_table["Segmentation"] = (
    results_seg.query("do_batch_normalization")
    .groupby(["normalization", "extended_experiment_type"])
    .Dice.mean()
    .dropna()
    .unstack(level=1)
    .round(3)
)

res_table["Sex"] = (
    data_class.query("task == 'sex'")
    .groupby(["normalization", "extended_experiment_type"])
    .auc_ovo.mean()
    .dropna()
    .unstack(level=1)
    .round(3)
)

res_table["Dworak"] = (
    data_class.query("task == 'dworak'")
    .groupby(["normalization", "extended_experiment_type"])
    .auc_ovo.mean()
    .dropna()
    .unstack(level=1)
    .round(3)
)

res_table["Age"] = (
    data_reg.groupby(["normalization", "extended_experiment_type"])
    .age_rmse.mean()
    .dropna()
    .unstack(level=1)
    .round(3)
)

styler = res_table.style
styler = styler.format(precision=3)
display_dataframe(styler)
# print(
#     styler.to_latex(
#         convert_css=True,
#         caption="Caption.",
#         label="results_table",
#     )
# )

# %%

print("Make a reduced table for the error without nBN without internal data")

results_seg = results_seg_new_names.query(
    "before_therapy & postprocessed & name != 'combined_models' & version == 'best'"
).copy()
data_class = results_class_task_new_names.query("version == 'best'").copy()
data_reg = results_reg_new_names.query("version == 'best'").copy()


res_table = pd.DataFrame(
    index=results_seg.normalization.cat.categories,
    columns=pd.MultiIndex.from_product(
        [
            ["Segmentation", "Sex", "Dworak", "Age"],
            [
                "All",
                "Except-One-Internal",
                "Except-One-External",
                "Single-Center-Internal",
                "Single-Center-External",
            ],
        ]
    ),
)

res_table["Segmentation"] = (
    results_seg.query("do_batch_normalization")
    .groupby(["normalization", "extended_experiment_type"])
    .Dice.sem()
    .dropna()
    .unstack(level=1)
    .round(3)
)

res_table["Sex"] = (
    data_class.query("task == 'sex'")
    .groupby(["normalization", "extended_experiment_type"])
    .auc_ovo.sem()
    .dropna()
    .unstack(level=1)
    .round(3)
)

res_table["Dworak"] = (
    data_class.query("task == 'dworak'")
    .groupby(["normalization", "extended_experiment_type"])
    .auc_ovo.sem()
    .dropna()
    .unstack(level=1)
    .round(3)
)

res_table["Age"] = (
    data_reg.groupby(["normalization", "extended_experiment_type"])
    .age_rmse.sem()
    .dropna()
    .unstack(level=1)
    .round(3)
)

styler = res_table.style
styler = styler.format(precision=3)
display_dataframe(styler)
# print(
#     styler.to_latex(
#         convert_css=True,
#         caption="Caption.",
#         label="results_table",
#     )
# )

# %%

print("Look for significance")

for experiment_type in ["All", "Except-One", "Single-Center"]:
    display_markdown(f"### {experiment_type}")

    data_seg = results_seg_new_names.query(
        "before_therapy & postprocessed & name != 'combined_models' & version == 'best'"
        f" & experiment_type == '{experiment_type}'"
    )
    if experiment_type == "All":
        grouped_data = data_seg.groupby(["do_batch_normalization", "normalization"])
        plot_significance(grouped_data, f"{experiment_type} - Segmentation", "Dice")
    else:
        for ext in ["not external", "external"]:
            data_seg_ext = data_seg.query(ext)
            grouped_data = data_seg_ext.groupby(["do_batch_normalization", "normalization"])
            plot_significance(
                grouped_data, f"{experiment_type} - {ext} - Segmentation", "Dice"
            )

    if experiment_type == "All":
        grouped_data = data_seg.query("do_batch_normalization").groupby("normalization")
        plot_significance(grouped_data, f"{experiment_type} - Segmentation - BN", "Dice")
    else:
        for ext in ["not external", "external"]:
            data_seg_ext = data_seg.query(ext)
            grouped_data = data_seg_ext.query("do_batch_normalization").groupby(
                "normalization"
            )
            plot_significance(
                grouped_data, f"{experiment_type} - {ext} - Segmentation - BN", "Dice"
            )

    if experiment_type == "All":
        grouped_data = data_seg.query("do_batch_normalization").groupby("normalization")
        plot_significance(grouped_data, f"{experiment_type} - Segmentation - BN", "Dice")
    else:
        for ext in ["not external", "external"]:
            data_seg_ext = data_seg.query(ext)
            grouped_data = data_seg_ext.query("not do_batch_normalization").groupby(
                "normalization"
            )
            plot_significance(
                grouped_data, f"{experiment_type} - {ext} - Segmentation - no BN", "Dice"
            )

    for task in good_classification_tasks:
        data_class = results_class_task_new_names.query(
            f"experiment_type == '{experiment_type}' & version == 'best' & task == '{task}'"
        )
        # only use first label to not overstate the significance
        if experiment_type == "All":
            grouped_data = data_class.groupby(["normalization"])
            plot_significance(grouped_data, f"{experiment_type} - {task}", "auc_ovo")
        else:
            for ext in ["not external", "external"]:
                grouped_data = data_class.query(ext).groupby(["normalization"])
                plot_significance(
                    grouped_data, f"{experiment_type} - {ext} - {task}", "auc_ovo"
                )

    data_reg = results_reg_new_names.query(
        f"experiment_type == '{experiment_type}' & version == 'best'"
    )
    if experiment_type == "All":
        grouped_data = data_reg.groupby(["normalization"])
        plot_significance(grouped_data, f"{experiment_type} - Age", "age_rmse")
    else:
        for ext in ["not external", "external"]:
            grouped_data = data_reg.query(ext).groupby(["normalization"])
            plot_significance(grouped_data, f"{experiment_type} - {ext} - Age", "age_rmse")


# %%

print("Acquisition parameters")

WIDTH = 193.44536 / 72

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(WIDTH, 0.84 * WIDTH))
hatched_histplot(
    data=acquisition_params.query("location != 'Mannheim-not-from-study'").replace(
        NEW_NAMES
    ),
    hue="location",
    x="pixel_spacing",
    multiple="stack",
    bins=np.arange(0.25, 1.66, 0.1),
    hue_order=CENTER_ORDER,
    legend=False,
    ax=axes,
)
plt.ylabel("count")
plt.xlabel("in-plane resolution (mm)")
plt.ylim(-4, 175)
save_pub("params_pixel_spacing", bbox_inches="tight")

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(WIDTH, 0.84 * WIDTH))
hatched_histplot(
    data=acquisition_params.query("location != 'Mannheim-not-from-study'").replace(
        NEW_NAMES
    ),
    hue="location",
    x="echo_time",
    multiple="stack",
    bins=np.arange(65, 226, 10),
    hue_order=CENTER_ORDER,
    legend=True,
    ax=axes,
)
plt.ylabel("count")
plt.xlabel("echo time (ms)")
plt.ylim(-4, 175)
save_pub("params_echo_time", bbox_inches="tight")

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(WIDTH, 0.84 * WIDTH))
hatched_histplot(
    data=acquisition_params.query("location != 'Mannheim-not-from-study'").replace(
        NEW_NAMES
    ),
    hue="location",
    x="flip_angle",
    multiple="stack",
    bins=np.arange(85, 186, 10),
    hue_order=CENTER_ORDER,
    legend=False,
    ax=axes,
)
plt.xlabel("flip angle (°)")
plt.ylabel("count")
plt.ylim(-4, 175)
save_pub("params_flip_angle", bbox_inches="tight")

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(WIDTH, 0.84 * WIDTH))
hatched_histplot(
    data=acquisition_params.query("location != 'Mannheim-not-from-study'").replace(
        NEW_NAMES
    ),
    hue="location",
    x="repetition_time",
    multiple="stack",
    bins=np.arange(500, 13000, 1000),
    hue_order=CENTER_ORDER,
    legend=False,
    ax=axes,
)
plt.xlabel("repetition time (ms)")
plt.ylabel("count")
plt.ylim(-4, 175)
save_pub("params_repetition_time", bbox_inches="tight")

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(WIDTH, 0.84 * WIDTH))
hatched_histplot(
    data=acquisition_params.query("location != 'Mannheim-not-from-study'").replace(
        NEW_NAMES
    ),
    hue="location",
    x="slice_thickness",
    multiple="stack",
    bins=np.arange(0.45, 6.05, 0.4),
    hue_order=CENTER_ORDER,
    legend=False,
    ax=axes,
)
plt.xlabel("slice thickness (mm)")
plt.ylabel("count")
save_pub("params_slice_thickness", bbox_inches="tight")

# %%

print("Acquisition parameters")

WIDTH = 193.44536 / 72


def plot_params(param):
    """Do the plots for the parameters"""
    _, axes_params = plt.subplots(nrows=1, ncols=1, figsize=(WIDTH, 0.84 * WIDTH))
    sns.boxplot(
        data=acquisition_params.replace(NEW_NAMES),
        x="location",
        y=param,
        ax=axes_params,
        medianprops={"color": "#21A6BFFF", "linestyle": "-"},
    )
    plt.xlabel("location")
    turn_ticks(axes_params)
    return axes_params


axes = plot_params("pixel_spacing")
plt.ylabel("in-plane resolution (mm)")
save_pub("params_pixel_spacing_box", bbox_inches=Bbox.from_extents(-0.4, -0.5, 2.5, 2.05))

axes = plot_params("echo_time")
plt.ylabel("echo time (ms)")
save_pub("params_echo_time_box", bbox_inches=Bbox.from_extents(-0.4, -0.5, 2.5, 2.05))

axes = plot_params("flip_angle")
plt.ylabel("flip angle (°)")
save_pub("params_flip_angle_box", bbox_inches=Bbox.from_extents(-0.4, -0.5, 2.5, 2.05))

axes = plot_params("repetition_time")
plt.ylabel("repetition time (ms)")
save_pub("params_repetition_time_box", bbox_inches=Bbox.from_extents(-0.4, -0.5, 2.5, 2.05))

axes = plot_params("slice_thickness")
plt.ylabel("slice thickness (mm)")
save_pub("params_slice_thickness_box", bbox_inches=Bbox.from_extents(-0.4, -0.5, 2.5, 2.05))


# %%

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(5.46, 2.6))

sns.histplot(
    data=acquisition_params.replace(NEW_NAMES),
    hue="location",
    x="pixel_spacing",
    multiple="stack",
    bins=np.arange(0.25, 1.66, 0.1),
    hue_order=CENTER_ORDER,
    ax=axes[0],
    legend=False,
)
axes[0].set_xlabel("in-plane resolution (mm)")
axes[0].set_ylim(-4, 250)

sns.histplot(
    data=acquisition_params.replace(NEW_NAMES),
    hue="location",
    x="echo_time",
    multiple="stack",
    bins=np.arange(65, 226, 10),
    hue_order=CENTER_ORDER,
    ax=axes[1],
    legend=True,
)
axes[1].set_xlabel("echo time (ms)")
axes[1].set_ylim(-4, 250)

axes[1].get_legend().set(bbox_to_anchor=[0.46, 0, 0, 1], title="Location")

titles = [
    "A",
    "B",
]
for ax, tlt in zip(axes, titles):
    ax.set_title(tlt)

plt.tight_layout()
save_pub("params", bbox_inches="tight")

fig, axes = plt.subplots(
    nrows=7, ncols=2, sharex="col", sharey="row", figsize=(3.7, 7 * 1.2)
)

for ax_line, center in zip(axes, CENTER_ORDER):
    sns.histplot(
        data=acquisition_params.replace(NEW_NAMES).query(f"location == '{center}'"),
        x="pixel_spacing",
        multiple="stack",
        bins=np.arange(0.25, 1.66, 0.1),
        ax=ax_line[0],
        legend=False,
    )
    ax_line[0].set_xlabel("in-plane resolution (mm)")

    sns.histplot(
        data=acquisition_params.replace(NEW_NAMES).query(f"location == '{center}'"),
        x="echo_time",
        multiple="stack",
        bins=np.arange(65, 226, 10),
        ax=ax_line[1],
        legend=False,
    )
    ax_line[1].set_xlabel("echo time (ms)")

    for ax in ax_line:
        ax.set_title(center)

plt.tight_layout()
save_pub("params_sep", bbox_inches="tight")

print(f"In-Plane Min: {acquisition_params.pixel_spacing.min():.2f} mm")
print(f"In-Plane Max: {acquisition_params.pixel_spacing.max():.2f} mm")

print(f"Echo Time Min: {acquisition_params.echo_time.min():.2f} ms")
print(f"Echo Time Max: {acquisition_params.echo_time.max():.2f} ms")

print(f"Flip Angle Min: {acquisition_params.flip_angle.min():.2f} °")
print(f"Flip Angle Max: {acquisition_params.flip_angle.max():.2f} °")

print(f"Repetition Time Min: {acquisition_params.repetition_time.min():.2f} ms")
print(f"Repetition Time Max: {acquisition_params.repetition_time.max():.2f} ms")

display_dataframe(
    pd.DataFrame(acquisition_params.echo_time.round(-1).value_counts()).sort_index()
)

display_dataframe(acquisition_params.replace(NEW_NAMES)[acquisition_params.echo_time > 200])

# %%

display_markdown("### Calculations for the paper")

seg_all = results_seg_new_names.query(
    "before_therapy & postprocessed & name != 'combined_models' & version == 'best'"
    " & experiment_type == 'All'"
)
dice_all_bn = seg_all.query("do_batch_normalization").Dice
print(f"Dice all BN mean:   {dice_all_bn.mean():.2f} ± {dice_all_bn.std():.2f}")
print(f"Dice all BN median: {dice_all_bn.median():.2f}")
print()

dice_all_no_bn = seg_all.query("not do_batch_normalization").Dice
print(f"Dice all no BN mean:   {dice_all_no_bn.mean():.2f} ± {dice_all_no_bn.std():.2f}")
print(f"Dice all no BN median: {dice_all_no_bn.median():.2f}")
no_bn_mean = (
    seg_all.query("not do_batch_normalization").groupby("normalization").Dice.mean()
)
print(f"Dice all no BN not working: {', '.join(no_bn_mean[no_bn_mean < 0.1].index)}")
print(f"Dice all no BN working seg {no_bn_mean[no_bn_mean > 0.1].mean():.2f}")

# %%

# set image parameters

STANDARD_PATIENT = "1031_1_l0_d0"
STANDARD_SLICES = {
    "T2w": 33,
    "b800": 19,
    "ADC": 19,
}

# %%

# images for the paper

for name, norm_method in zip(
    ["perc-hm", "hm"],
    [
        normalization.HMQuantile,
        normalization.HistogramMatching,
    ],
):

    MAX_VALUE = 600

    image_path = orig_dataset[STANDARD_PATIENT]["images"][0]
    image_sitk = sitk.ReadImage(str(data_dir / image_path))
    norm_dir_hist = (
        experiment_dir / "Normalization_all" / "data_preprocessed" / norm_method.enum.name
    )
    norm_file = norm_dir_hist / "normalization_mod0.yaml"
    norm = norm_method.from_file(norm_file)
    image_normed = norm.normalize(image_sitk)

    FIG_WIDTH = 6
    FIG_HEIGHT = 1.8
    widths = [0.3, 0.3, 0.2, 0.2]
    paddings = [0.3, 0.3, 0.1, 0.1]
    IMG_WIDTH = FIG_WIDTH * widths[-1] * (1 - paddings[-1])
    heights = [IMG_WIDTH / FIG_HEIGHT] * 4
    rects = []
    OFFSET = -0.02
    for w, h, p in zip(widths, heights, paddings):
        rects.append([w * p + OFFSET, (1 - h) / 2, w * (1 - p), h])
        OFFSET += w
    figure = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    axes = []
    for r in rects:
        ax = plt.Axes(figure, r)
        figure.add_axes(ax)
        axes.append(ax)

    if name == "hm":
        landmarks = norm.get_landmarks(image_sitk)[0]
    else:
        quant = normalization.Quantile(
            lower_q=norm.quantiles[0], upper_q=norm.quantiles[-1]
        )
        landmarks = norm.get_landmarks(quant.normalize(image_sitk))[0]
    axes[0].plot(norm.quantiles * 100, landmarks, label="orig.")
    axes[0].plot(norm.quantiles * 100, norm.standard_scale, label="norm.")
    axes[0].set_ylabel("Intensity")
    axes[0].set_xlim((0, 100))
    axes[0].set_xticks(norm.quantiles * 100)
    axes[0].set_xlabel("Percentile")
    axes[0].legend()
    axes[0].set_title("Landmarks")
    axes[0].yaxis.labelpad = 0
    axes[0].tick_params(axis="y", pad=1)
    axes[0].tick_params(axis="x", pad=1)
    if name == "perc-hm":
        axes[0].set_ylim(-1.05, 1.05)

    img_np = sitk.GetArrayFromImage(image_sitk)
    img_flat = 2 * img_np.reshape(-1) / MAX_VALUE - 1
    img_norm_np = sitk.GetArrayFromImage(image_normed)
    image_norm_flat = img_norm_np.reshape(-1)
    dataframe = pd.DataFrame(
        {
            "Intensity": np.concatenate((img_flat, image_norm_flat)),
            "image": ["Orig."] * len(img_flat) + ["Norm."] * len(image_norm_flat),
        },
    )

    ax_no_norm = axes[1].twiny()
    sns.histplot(
        data=dataframe,
        x="Intensity",
        hue="image",
        bins=np.arange(-1.05, 1.1, 0.1),
        stat="proportion",
        ax=axes[1],
        legend=False,
    )

    axes[1].set_xlim((-1.05, 1.05))
    axes[1].yaxis.labelpad = 1
    axes[1].tick_params(axis="y", pad=1)
    axes[1].tick_params(axis="x", pad=1)
    axes[1].tick_params(top=False)
    axes[1].set_xlabel("Intensity Norm.")
    axes[1].grid()

    ax_no_norm.set_xlim((-5, MAX_VALUE + MAX_VALUE / 20))
    ax_no_norm.set_xlabel("Intensity Orig.")
    ax_no_norm.set_ylabel("Proportion")
    ax_no_norm.yaxis.labelpad = 1
    ax_no_norm.tick_params(axis="y", pad=1)
    ax_no_norm.tick_params(axis="x", pad=1)
    ax_no_norm.tick_params(bottom=False)

    axes[2].imshow(
        img_np[STANDARD_SLICES["T2w"]],
        interpolation="nearest",
        cmap="gray",
        vmin=img_np[STANDARD_SLICES["T2w"]].min(),
        vmax=img_np[STANDARD_SLICES["T2w"]].max(),
        aspect="auto",
    )
    axes[2].axis("off")
    axes[2].set_title("Original Image")

    axes[3].imshow(
        img_norm_np[STANDARD_SLICES["T2w"]],
        interpolation="nearest",
        cmap="gray",
        vmin=img_norm_np[STANDARD_SLICES["T2w"]].min(),
        vmax=img_norm_np[STANDARD_SLICES["T2w"]].max(),
        aspect="auto",
    )
    axes[3].axis("off")
    axes[3].set_title("Normalized Image")

    save_pub(f"{name}-paper-hist_plot", bbox_inches="tight")

image_path = orig_dataset[STANDARD_PATIENT]["images"][0]
image_sitk = sitk.ReadImage(str(data_dir / image_path))
img_np = sitk.GetArrayFromImage(image_sitk)
img_flat = img_np.reshape(-1)
slice_original = img_np[STANDARD_SLICES["T2w"]]
dataframes = []
slices_normed = []

for name, norm_method in zip(
    [
        "perc",
        "hm",
        "perc-hm",
        "mean-std",
        "window",
        "GAN-Seg",
        "GAN-Def",
        "GAN-Img",
        "GAN-Win",
        "GAN-No-ed",
    ],
    [
        normalization.Quantile,
        normalization.HistogramMatching,
        normalization.HMQuantile,
        normalization.MeanSTD,
        normalization.Window,
        GanDiscriminators,
        GanDiscriminators,
        GanDiscriminators,
        GanDiscriminators,
        GanDiscriminators,
    ],
):

    print(name)
    if name == "GAN-Def":
        norm_dir_method = (
            experiment_dir
            / "Normalization_all"
            / "data_preprocessed"
            / "GAN_DISCRIMINATORS_3_64_0.50_BetterConv_0.00001"
        )
    elif name == "GAN-Seg":
        norm_dir_method = (
            experiment_dir
            / "Normalization_all"
            / "data_preprocessed"
            / "GAN_DISCRIMINATORS_3_64_0.50_BetterConv_0.00001_seg"
        )
    elif name == "GAN-Img":
        norm_dir_method = (
            experiment_dir
            / "Normalization_all"
            / "data_preprocessed"
            / "GAN_DISCRIMINATORS_3_64_0.50_BetterConv_0.00001_all_image"
        )
    elif name == "GAN-Win":
        norm_dir_method = (
            experiment_dir
            / "Normalization_all"
            / "data_preprocessed"
            / "GAN_DISCRIMINATORS_3_64_0.50_BetterConv_0.00001_WINDOW"
        )
    elif name == "GAN-No-ed":
        norm_dir_method = (
            experiment_dir
            / "Normalization_all"
            / "data_preprocessed"
            / "GAN_DISCRIMINATORS_3_64_n-skp_BetterConv_0.00001_WINDOW"
        )
    elif name in ("hm", "perc-hm"):
        norm_dir_method = (
            experiment_dir
            / "Normalization_all"
            / "data_preprocessed"
            / norm_method.enum.name
        )
    else:
        norm_dir_method = experiment_dir / "data_preprocessed" / norm_method.enum.name
    norm_file = norm_dir_method / "normalization_mod0.yaml"
    norm = norm_method.from_file(norm_file)
    if name == "perc":
        norm_quant = norm
    if name == "window":
        norm_win = norm
    if name in ("GAN-Def", "GAN-Seg", "GAN-Img"):
        image_normed = norm.normalize(norm_quant.normalize(image_sitk))
    elif name in ("GAN-Win", "GAN-No-ed"):
        image_normed = norm.normalize(norm_win.normalize(image_sitk))
    else:
        image_normed = norm.normalize(image_sitk)

    OFFSET = -0.02
    widths = [0.3, 0.3, 0.25, 0.25]
    paddings = [0.15, 0.15, 0.1, 0.1]
    IMG_WIDTH = FIG_WIDTH * widths[-1] * (1 - paddings[-1])
    heights = [IMG_WIDTH / FIG_HEIGHT] * 4
    rects = []
    for w, h, p in zip(widths, heights, paddings):
        rects.append([w * p + OFFSET, (1 - h) / 2, w * (1 - p), h])
        OFFSET += w
    figure = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    axes = []
    for r in rects:
        ax = plt.Axes(figure, r)
        figure.add_axes(ax)
        axes.append(ax)

    img_norm_np = sitk.GetArrayFromImage(image_normed)
    image_norm_flat = img_norm_np.reshape(-1)
    dataframe = pd.DataFrame(
        {
            "Intensity": np.concatenate((img_flat, image_norm_flat)),
            "image": ["original image"] * len(img_flat)
            + ["normalized image"] * len(image_norm_flat),
        },
    )
    dataframes.append(dataframe)

    sns.histplot(
        data=dataframe.query("image == 'original image'"),
        x="Intensity",
        hue="image",
        bins=20,
        stat="proportion",
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title("Original Histogram")

    sns.histplot(
        data=dataframe.query("image == 'normalized image'"),
        x="Intensity",
        hue="image",
        bins=20,
        stat="proportion",
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Normalized Histogram")
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])

    axes[2].imshow(
        slice_original,
        interpolation="nearest",
        cmap="gray",
        vmin=img_np[STANDARD_SLICES["T2w"]].min(),
        vmax=img_np[STANDARD_SLICES["T2w"]].max(),
        aspect="auto",
    )
    axes[2].axis("off")
    axes[2].set_title("Original Image")

    axes[3].imshow(
        img_norm_np[STANDARD_SLICES["T2w"]],
        interpolation="nearest",
        cmap="gray",
        vmin=img_norm_np[STANDARD_SLICES["T2w"]].min(),
        vmax=img_norm_np[STANDARD_SLICES["T2w"]].max(),
        aspect="auto",
    )
    axes[3].axis("off")
    axes[3].set_title("Normalized Image")

    slices_normed.append(img_norm_np[STANDARD_SLICES["T2w"]])

    max_y = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(0, max_y)
    axes[1].set_ylim(0, max_y)

    save_pub(f"{name}-paper", bbox_inches="tight")

# %%
# make an image containing all normalizations
# fig, axes = plt.subplots(ncols=6, nrows=3, figsize=(7, 4), layout="constrained")

fig = plt.figure(figsize=(0.8, 0.8))
axes = []

bottom = 0
height_diff = 0.6
for row in range(3):
    ax_line = []
    axes.append(ax_line)
    left = 0
    for col in range(6):
        height = 1
        padding = (1 - (col + 1) // 3) * 0.0 + 0.1
        width = 1
        # (left, bottom, width, height)
        ax = plt.Axes(fig, [left, bottom, width, height])
        left += width + padding
        fig.add_axes(ax)
        ax_line.append(ax)
    bottom -= height + height_diff
axes = np.array(axes)

names = [
    "Perc",
    "HM",
    "Perc-HM",
    "M-STD",
    "Win",
    "GAN-Seg",
    "GAN-Def",
    "GAN-Img",
    "GAN-Win",
]

sns.histplot(
    data=dataframe.query("image == 'original image'"),
    x="Intensity",
    hue="image",
    bins=20,
    stat="proportion",
    ax=axes[0, 0],
    legend=False,
)
axes[0, 0].set_title("Original Histogram", fontsize=8)
axes[0, 0].set_ylim(0, 0.4)
axes[0, 0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
axes[0, 0].set_xticks([0, 500, 1000])

axes[0, 3].imshow(
    slice_original,
    interpolation="nearest",
    cmap="gray",
    vmin=slice_original.min(),
    vmax=slice_original.max(),
    aspect="equal",
)
axes[0, 3].set_title("Original Image", fontsize=8)

for ax, df, n in zip(axes[1:, :3].flat, dataframes, names):
    if n not in ("M-STD",):  # "Win"):
        bins = np.arange(-1.05, 1.06, 0.1)
    else:
        bins = 21
    sns.histplot(
        data=df.query("image == 'normalized image'"),
        x="Intensity",
        hue="image",
        bins=bins,
        stat="proportion",
        ax=ax,
        legend=False,
    )
    ax.set_title(n, fontsize=8)
    ax.set_ylim(0, 0.4)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    if n not in ("M-STD"):
        ax.set_xlim(-1.15, 1.15)

for ax, slc, n in zip(axes[1:, 3:].flat, slices_normed, names):
    ax.imshow(
        slc,
        interpolation="nearest",
        cmap="gray",
        vmin=slc.min(),
        vmax=slc.max(),
        aspect="equal",
    )
    ax.set_title(n, fontsize=8)

axes[0, 1].axis("off")
axes[0, 2].axis("off")
for ax in axes[:, 3:].flat:
    ax.axis("off")

for ax in axes[-1, :3]:
    ax.set_xlabel("Intensity")
for ax in axes[:-1, :3].flat:
    ax.set_xlabel("")

for ax in axes[:, 0]:
    ax.set_ylabel("Proportion")
for ax in axes[:, 1:].flat:
    ax.set_ylabel("")
    ax.set_yticklabels([])
    ax.tick_params(axis="y", which="both", length=0)

save_pub("norm-summary-paper", bbox_inches="tight")

# %%
# make an image containing all normalizations
# fig, axes = plt.subplots(ncols=6, nrows=3, figsize=(7, 4), layout="constrained")

fig = plt.figure(figsize=(0.8, 0.8))
axes = []

bottom = 0
height_diff = 0.6
for row in range(4):
    ax_line = []
    axes.append(ax_line)
    left = 0
    for col in range(4):
        height = 1
        padding = (1 - (col + 1) // 3) * 0.0 + 0.1
        width = 1
        # (left, bottom, width, height)
        ax = plt.Axes(fig, [left, bottom, width, height])
        left += width + padding
        fig.add_axes(ax)
        ax_line.append(ax)
    bottom -= height + height_diff
axes = np.array(axes)

names = [
    "Perc",
    "HM",
    "Perc-HM",
    "M-STD",
    "Win",
    "GAN-Seg",
    "GAN-Def",
    "GAN-Img",
    "GAN-Win",
]

sns.histplot(
    data=dataframe.query("image == 'original image'"),
    x="Intensity",
    hue="image",
    bins=20,
    stat="proportion",
    ax=axes[0, 0],
    legend=False,
)
axes[0, 0].set_title("Original Histogram", fontsize=8)
axes[0, 0].set_ylim(0, 0.4)
axes[0, 0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
axes[0, 0].set_xticks([0, 500, 1000])

axes[0, 3].imshow(
    slice_original,
    interpolation="nearest",
    cmap="gray",
    vmin=slice_original.min(),
    vmax=slice_original.max(),
    aspect="equal",
)
axes[0, 3].set_title("Original Image", fontsize=8)

for ax, df, n in zip(axes[1:, :2].flat, dataframes, names):
    if n not in ("M-STD",):  # "Win"):
        bins = np.arange(-1.05, 1.06, 0.1)
    else:
        bins = 21
    sns.histplot(
        data=df.query("image == 'normalized image'"),
        x="Intensity",
        hue="image",
        bins=bins,
        stat="proportion",
        ax=ax,
        legend=False,
    )
    ax.set_title(n, fontsize=8)
    ax.set_ylim(0, 0.4)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    if n not in ("M-STD"):
        ax.set_xlim(-1.15, 1.15)

for ax, slc, n in zip(axes[1:, 2:].flat, slices_normed, names):
    ax.imshow(
        slc,
        interpolation="nearest",
        cmap="gray",
        vmin=slc.min(),
        vmax=slc.max(),
        aspect="equal",
    )
    ax.set_title(n, fontsize=8)

axes[0, 1].axis("off")
axes[0, 2].axis("off")
for ax in axes[:, 2:].flat:
    ax.axis("off")

for ax in axes[-1, :2]:
    ax.set_xlabel("Intensity")
for ax in axes[:-1, :2].flat:
    ax.set_xlabel("")

for ax in axes[:, 0]:
    ax.set_ylabel("Proportion")
for ax in axes[:, 1:].flat:
    ax.set_ylabel("")
    ax.set_yticklabels([])
    ax.tick_params(axis="y", which="both", length=0)

save_pub("norm-summary-paper", bbox_inches="tight")

# %%
# make a nice description of the hm methods
images = orig_dataset[STANDARD_PATIENT]["images"]
images_sitk = [sitk.ReadImage(str(data_dir / img)) for img in images]
norm_dir_hist = (
    experiment_dir / "Normalization_all" / "data_preprocessed" / "HISTOGRAM_MATCHING"
)
norm_files = [norm_dir_hist / f"normalization_mod{i}.yaml" for i in range(3)]
norms = [normalization.HistogramMatching.from_file(f) for f in norm_files]
images_normed = [n.normalize(img) for img, n in zip(images_sitk, norms)]

axes = create_axes()

for i, (n, lbl, img, img_norm, ax_line) in enumerate(
    zip(norms, ["T2w", "b800", "ADC"], images_sitk, images_normed, axes)
):
    ax_line[0].plot(n.quantiles * 100, n.standard_scale, label=f"{lbl} - std.")
    ax_line[0].plot(n.quantiles * 100, n.get_landmarks(img)[0], label=f"{lbl} - img.")
    ax_line[0].set_ylabel("Intensity")
    ax_line[0].set_xlim((0, 100))
    ax_line[0].set_xticks(norms[0].quantiles * 100)
    ax_line[0].legend()
    if i == 2:
        ax_line[0].set_xlabel("Percentile")
    else:
        ax_line[0].axes.xaxis.set_ticklabels([])
        ax_line[0].axes.set_xlabel(None)
    if i == 0:
        ax_line[0].set_title("Landmarks")

    img_np = sitk.GetArrayFromImage(img)
    img_np = 2 * img_np / img_np.max() - 1
    img_flat = img_np.reshape(-1)
    img_norm_np = sitk.GetArrayFromImage(img_norm)
    image_norm_flat = img_norm_np.reshape(-1)
    dataframe = pd.DataFrame(
        {
            "Intensity": np.concatenate((img_flat, image_norm_flat)),
            "image": ["ori. img."] * len(img_flat) + ["norm. img."] * len(image_norm_flat),
        },
    )
    sns.histplot(
        data=dataframe,
        x="Intensity",
        hue="image",
        bins=20,
        stat="proportion",
        ax=ax_line[1],
        legend=i == 0,
    )
    if i == 0:
        ax_line[1].set_title("Histogram")
    if i != 2:
        ax_line[1].axes.set_xlabel(None)
    ax_line[1].set_xlim((-1, 1))

    ax_line[2].imshow(
        img_np[STANDARD_SLICES[lbl]], interpolation="nearest", cmap="gray", vmin=-1, vmax=1
    )
    ax_line[2].axis("off")
    if i == 0:
        ax_line[2].set_title("Original Image")

    ax_line[3].imshow(
        img_norm_np[STANDARD_SLICES[lbl]],
        interpolation="nearest",
        cmap="gray",
        vmin=-1,
        vmax=1,
    )
    ax_line[3].axis("off")
    if i == 0:
        ax_line[3].set_title("Normalized Image")

# (x_min, y_min, x_max, y_max)
save_pub("hm", bbox_inches=Bbox.from_extents(-0.8, 1.9, 10.5, 8.3))

# %%
# make a nice description of the window method
images = orig_dataset[STANDARD_PATIENT]["images"]
images_sitk = [sitk.ReadImage(str(data_dir / img)) for img in images]
norm_dir_hist = experiment_dir / "data_preprocessed" / "WINDOW"
norm_files = [norm_dir_hist / f"normalization_mod{i}.yaml" for i in range(3)]
norms = [normalization.Window.from_file(f) for f in norm_files]
for n in norms:
    n.clip_outliers = False
images_normed = [n.normalize(img) for img, n in zip(images_sitk, norms)]
axes = create_axes()
for i, (n, lbl, img, img_norm, ax_line) in enumerate(
    zip(norms, ["T2w", "b800", "ADC"], images_sitk, images_normed, axes)
):
    img_np = sitk.GetArrayFromImage(img)
    img_flat = img_np.reshape(-1)
    img_norm_np = sitk.GetArrayFromImage(img_norm)
    image_norm_flat = img_norm_np.reshape(-1)
    dataframe = pd.DataFrame(
        {
            "Intensity": np.concatenate((img_flat, image_norm_flat)),
            "image": ["original image"] * len(img_flat)
            + ["normalized image"] * len(image_norm_flat),
        },
    )

    sns.histplot(
        data=dataframe.query("image == 'original image'"),
        x="Intensity",
        hue="image",
        bins=20,
        stat="proportion",
        ax=ax_line[0],
        legend=False,
    )
    if i == 0:
        ax_line[0].set_title("Original Histogram")
    if i != 2:
        ax_line[0].axes.set_xlabel(None)

    MIN_B = -1.0
    MAX_B = 1.0
    BIN_WIDTH = (MAX_B - MIN_B) / 20
    MIN_DISP = MIN_B - BIN_WIDTH / 2
    MAX_DISP = MAX_B + BIN_WIDTH / 2
    sns.histplot(
        data=dataframe.query("image == 'normalized image'"),
        x="Intensity",
        hue="image",
        bins=np.linspace(MIN_DISP, MAX_DISP, 21),
        stat="proportion",
        ax=ax_line[1],
        legend=False,
    )
    if i == 0:
        ax_line[1].set_title("Normalized Histogram")
    ax_line[1].set_xlim((MIN_DISP - BIN_WIDTH, MAX_DISP + BIN_WIDTH))
    if i != 2:
        ax_line[1].axes.set_xlabel(None)
    ax_line[0].set_ylim(ax_line[1].get_ylim())

    ax_line[2].imshow(
        img_np[STANDARD_SLICES[lbl]],
        interpolation="nearest",
        cmap="gray",
        vmin=img_np.min(),
        vmax=img_np.max(),
    )
    ax_line[2].axis("off")
    if i == 0:
        ax_line[2].set_title("Original Image")

    ax_line[3].imshow(
        img_norm_np[STANDARD_SLICES[lbl]],
        interpolation="nearest",
        cmap="gray",
        vmin=img_norm_np.min(),
        vmax=img_norm_np.max(),
    )
    ax_line[3].axis("off")
    if i == 0:
        ax_line[3].set_title("Normalized Image")

save_pub("window", bbox_inches=Bbox.from_extents(-0.8, 1.9, 10.5, 8.3))

# %%
# make a nice description of the mean-std method
images = orig_dataset[STANDARD_PATIENT]["images"]
images_sitk = [sitk.ReadImage(str(data_dir / img)) for img in images]
norm_dir_hist = experiment_dir / "data_preprocessed" / "MEAN_STD"
norm_files = [norm_dir_hist / f"normalization_mod{i}.yaml" for i in range(3)]
norms = [normalization.MeanSTD.from_file(f) for f in norm_files]
images_normed = [n.normalize(img) for img, n in zip(images_sitk, norms)]
axes = create_axes()
for i, (n, lbl, img, img_norm, ax_line) in enumerate(
    zip(norms, ["T2w", "b800", "ADC"], images_sitk, images_normed, axes)
):
    img_np = sitk.GetArrayFromImage(img)
    img_flat = img_np.reshape(-1)
    img_norm_np = sitk.GetArrayFromImage(img_norm)
    image_norm_flat = img_norm_np.reshape(-1)
    dataframe = pd.DataFrame(
        {
            "Intensity": np.concatenate((img_flat, image_norm_flat)),
            "image": ["original image"] * len(img_flat)
            + ["normalized image"] * len(image_norm_flat),
        },
    )

    sns.histplot(
        data=dataframe.query("image == 'original image'"),
        x="Intensity",
        hue="image",
        bins=20,
        stat="proportion",
        ax=ax_line[0],
        legend=False,
    )
    if i == 0:
        ax_line[0].set_title("Original Histogram")
    if i != 2:
        ax_line[0].axes.set_xlabel(None)

    MIN_B = -1.6
    MAX_B = 2.5
    BIN_WIDTH = (MAX_B - MIN_B) / 20
    MIN_DISP = MIN_B - BIN_WIDTH / 2
    MAX_DISP = MAX_B + BIN_WIDTH / 2
    sns.histplot(
        data=dataframe.query("image == 'normalized image'"),
        x="Intensity",
        hue="image",
        bins=np.linspace(MIN_DISP, MAX_DISP, 21),
        stat="proportion",
        ax=ax_line[1],
        legend=False,
    )
    if i == 0:
        ax_line[1].set_title("Normalized Histogram")
    ax_line[1].set_xlim((MIN_DISP - BIN_WIDTH, MAX_DISP + BIN_WIDTH))
    if i != 2:
        ax_line[1].axes.set_xlabel(None)
    ax_line[1].set_ylim(ax_line[0].get_ylim())

    ax_line[2].imshow(
        img_np[STANDARD_SLICES[lbl]],
        interpolation="nearest",
        cmap="gray",
        vmin=img_np.min(),
        vmax=img_np.max(),
    )
    ax_line[2].axis("off")
    if i == 0:
        ax_line[2].set_title("Original Image")

    ax_line[3].imshow(
        img_norm_np[STANDARD_SLICES[lbl]],
        interpolation="nearest",
        cmap="gray",
        vmin=img_norm_np.min(),
        vmax=img_norm_np.max(),
    )
    ax_line[3].axis("off")
    if i == 0:
        ax_line[3].set_title("Normalized Image")

save_pub("mean-std", bbox_inches=Bbox.from_extents(-0.8, 1.9, 10.5, 8.3))

# %%
# make a nice description of the perc method
images = orig_dataset[STANDARD_PATIENT]["images"]
images_sitk = [sitk.ReadImage(str(data_dir / img)) for img in images]
norm_dir_hist = experiment_dir / "data_preprocessed" / "QUANTILE"
norm_files = [norm_dir_hist / f"normalization_mod{i}.yaml" for i in range(3)]
norms = [normalization.Quantile.from_file(f) for f in norm_files]
images_normed = [n.normalize(img) for img, n in zip(images_sitk, norms)]
axes = create_axes()
for i, (n, lbl, img, img_norm, ax_line) in enumerate(
    zip(norms, ["T2w", "b800", "ADC"], images_sitk, images_normed, axes)
):
    img_np = sitk.GetArrayFromImage(img)
    img_flat = img_np.reshape(-1)
    img_norm_np = sitk.GetArrayFromImage(img_norm)
    image_norm_flat = img_norm_np.reshape(-1)
    dataframe = pd.DataFrame(
        {
            "Intensity": np.concatenate((img_flat, image_norm_flat)),
            "image": ["original image"] * len(img_flat)
            + ["normalized image"] * len(image_norm_flat),
        },
    )

    sns.histplot(
        data=dataframe.query("image == 'original image'"),
        x="Intensity",
        hue="image",
        bins=20,
        stat="proportion",
        ax=ax_line[0],
        legend=False,
    )
    if i == 0:
        ax_line[0].set_title("Original Histogram")
    if i != 2:
        ax_line[0].axes.set_xlabel(None)

    sns.histplot(
        data=dataframe.query("image == 'normalized image'"),
        x="Intensity",
        hue="image",
        bins=np.linspace(-1.05, 1.05, 21),
        stat="proportion",
        ax=ax_line[1],
        legend=False,
    )
    if i == 0:
        ax_line[1].set_title("Normalized Histogram")
    if i != 2:
        ax_line[1].axes.xaxis.set_ticklabels([])
        ax_line[1].axes.set_xlabel(None)
    ax_line[1].set_xlim((-1.15, 1.15))
    # remove y-axis
    ax_line[1].set_ylim(ax_line[0].get_ylim())
    if i != 2:
        ax_line[1].axes.set_xlabel(None)

    ax_line[2].imshow(
        img_np[STANDARD_SLICES[lbl]],
        interpolation="nearest",
        cmap="gray",
        vmin=img_np.min(),
        vmax=img_np.max(),
    )
    ax_line[2].axis("off")
    if i == 0:
        ax_line[2].set_title("Original Image")

    ax_line[3].imshow(
        img_norm_np[STANDARD_SLICES[lbl]],
        interpolation="nearest",
        cmap="gray",
        vmin=img_norm_np.min(),
        vmax=img_norm_np.max(),
    )
    ax_line[3].axis("off")
    if i == 0:
        ax_line[3].set_title("Normalized Image")

save_pub("perc", bbox_inches=Bbox.from_extents(-0.8, 1.9, 10.5, 8.3))

# %%
# make a nice description of the perc-hm methods
images = orig_dataset[STANDARD_PATIENT]["images"]
images_sitk = [sitk.ReadImage(str(data_dir / img)) for img in images]
norm_dir_hist = experiment_dir / "Normalization_all" / "data_preprocessed" / "HM_QUANTILE"
norm_files = [norm_dir_hist / f"normalization_mod{i}.yaml" for i in range(3)]
norms = [normalization.HMQuantile.from_file(f) for f in norm_files]
images_normed = [n.normalize(img) for img, n in zip(images_sitk, norms)]
axes = create_axes()
for i, (n, lbl, img, img_norm, ax_line) in enumerate(
    zip(norms, ["T2w", "b800", "ADC"], images_sitk, images_normed, axes)
):
    ax_line[0].plot(n.quantiles * 100, n.standard_scale, label=f"{lbl} - std.")
    quant = normalization.Quantile(lower_q=n.quantiles[0], upper_q=n.quantiles[-1])
    landmarks = n.get_landmarks(quant.normalize(img))[0]
    landmarks = landmarks / landmarks.max()
    ax_line[0].plot(n.quantiles * 100, landmarks, label=f"{lbl} - img.")
    ax_line[0].set_ylabel("Intensity")
    ax_line[0].set_xlim((0, 100))
    ax_line[0].set_xticks(norms[0].quantiles * 100)
    if i == 2:
        ax_line[0].set_xlabel("Percentile")
    else:
        ax_line[0].axes.set_xlabel(None)
    if i == 0:
        ax_line[0].legend()
        ax_line[0].set_title("Landmarks")

    img_np = sitk.GetArrayFromImage(img)
    img_np = 2 * img_np / img_np.max() - 1
    img_flat = img_np.reshape(-1)
    img_norm_np = sitk.GetArrayFromImage(img_norm)
    image_norm_flat = img_norm_np.reshape(-1)
    dataframe = pd.DataFrame(
        {
            "Intensity": np.concatenate((img_flat, image_norm_flat)),
            "image": ["orig. img."] * len(img_flat) + ["norm. img."] * len(image_norm_flat),
        },
    )
    sns.histplot(
        data=dataframe,
        x="Intensity",
        hue="image",
        bins=20,
        stat="proportion",
        ax=ax_line[1],
        legend=i == 0,
    )
    if i == 0:
        ax_line[1].set_title("Histogram")
    ax_line[1].set_xlim((-1, 1))
    if i != 2:
        ax_line[1].axes.set_xlabel(None)

    ax_line[2].imshow(
        img_np[STANDARD_SLICES[lbl]],
        interpolation="nearest",
        cmap="gray",
        vmin=-1,
        vmax=1,
    )
    ax_line[2].axis("off")
    if i == 0:
        ax_line[2].set_title("Original Image")

    ax_line[3].imshow(
        img_norm_np[STANDARD_SLICES[lbl]],
        interpolation="nearest",
        cmap="gray",
        vmin=-1,
        vmax=1,
    )
    ax_line[3].axis("off")
    if i == 0:
        ax_line[3].set_title("Normalized Image")

save_pub("perc-hm", bbox_inches=Bbox.from_extents(-0.8, 1.9, 10.5, 8.3))

# %%
# make a nice description of the GAN-Def methods
image = sitk.ReadImage(
    str(
        experiment_dir
        / "data_preprocessed"
        / "QUANTILE"
        / f"sample-{STANDARD_PATIENT}.nii.gz"
    )
)
images_sitk = [sitk.VectorIndexSelectionCast(image, i) for i in range(3)]
norm_dir_hist = (
    experiment_dir
    / "Normalization_all"
    / "data_preprocessed"
    / "GAN_DISCRIMINATORS_3_64_0.50_BetterConv_0.00001"
)
norm_files = [norm_dir_hist / f"normalization_mod{i}.yaml" for i in range(3)]
norms = [GanDiscriminators.from_file(f) for f in norm_files]
images_normed = [n.normalize(img) for img, n in zip(images_sitk, norms)]

axes = create_axes()
for i, (n, lbl, img, img_norm, ax_line) in enumerate(
    zip(norms, ["T2w", "b800", "ADC"], images_sitk, images_normed, axes)
):
    # ax_line[0].plot(n.quantiles * 100, n.standard_scale, label=f"{lbl} - std.")
    # quant = normalization.Quantile(lower_q=n.quantiles[0], upper_q=n.quantiles[-1])
    # landmarks = n.get_landmarks(quant.normalize(img))[0]
    # landmarks = landmarks / landmarks.max()
    # ax_line[0].plot(n.quantiles * 100, landmarks, label=f"{lbl} - img.")
    # ax_line[0].set_ylabel("Intensity")
    # ax_line[0].set_xlim((0, 100))
    # ax_line[0].set_xticks(norms[0].quantiles * 100)
    # if i == 2:
    #     ax_line[0].set_xlabel("Percentile")
    # else:
    #     ax_line[0].axes.set_xlabel(None)
    # if i == 0:
    #     ax_line[0].legend()
    #     ax_line[0].set_title("Landmarks")

    img_np = sitk.GetArrayFromImage(img)
    img_np = 2 * img_np / img_np.max() - 1
    img_flat = img_np.reshape(-1)
    img_norm_np = sitk.GetArrayFromImage(img_norm)
    image_norm_flat = img_norm_np.reshape(-1)
    dataframe = pd.DataFrame(
        {
            "Intensity": np.concatenate((img_flat, image_norm_flat)),
            "image": ["orig. img."] * len(img_flat) + ["norm. img."] * len(image_norm_flat),
        },
    )
    sns.histplot(
        data=dataframe,
        x="Intensity",
        hue="image",
        bins=20,
        stat="proportion",
        ax=ax_line[1],
        legend=i == 0,
    )
    if i == 0:
        ax_line[1].set_title("Histogram")
    ax_line[0].set_xlim((-1, 1))
    if i != 2:
        ax_line[0].axes.set_xlabel(None)

    ax_line[2].imshow(
        img_np[STANDARD_SLICES[lbl]], interpolation="nearest", cmap="gray", vmin=-1, vmax=1
    )
    ax_line[2].axis("off")
    if i == 0:
        ax_line[2].set_title("Original Image")

    ax_line[3].imshow(
        img_norm_np[STANDARD_SLICES[lbl]],
        interpolation="nearest",
        cmap="gray",
        vmin=-1,
        vmax=1,
    )
    ax_line[3].axis("off")
    if i == 0:
        ax_line[3].set_title("Normalized Image")

save_pub("GAN-Def", bbox_inches=Bbox.from_extents(-0.8, 1.9, 10.5, 8.3))

# %%
# Make a graph of the patients treatment intervals

patient_data = pd.read_csv(data_dir / "patients.csv", sep=";", index_col=0).infer_objects()
dt_columns = ["birthday"] + [col for col in patient_data if "date" in col]

# correct type
for col in dt_columns:
    patient_data[col] = pd.to_datetime(patient_data[col])

patient_data["from_study"] = True
patient_data.loc[patient_data.location == "Mannheim-not-from-study", "from_study"] = False
patient_data["location"] = patient_data.location.replace(NEW_NAMES)

# %%

patient_data["mri_1_to_op"] = (
    patient_data.OP_date - patient_data.date_before_therapy_MRI
).dt.days
patient_data["mri_2_to_op"] = (
    patient_data.OP_date - patient_data.date_after_therapy_MRI
).dt.days
patient_data["mri_1_to_mri_2"] = (
    patient_data.date_after_therapy_MRI - patient_data.date_before_therapy_MRI
).dt.days

split_bars = True

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(WIDTH, 0.84 * WIDTH))
patient_data["date_before_therapy_MRI_frac"] = (
    patient_data.date_before_therapy_MRI.dt.year
    + patient_data.date_before_therapy_MRI.dt.day_of_year / 365
)
hatched_histplot(
    data=patient_data[patient_data.date_before_therapy_MRI_frac.notna()],
    hue="from_study",
    x="date_before_therapy_MRI_frac",
    multiple="layer",
    bins=np.arange(2009.75, 2019, 0.5),
    legend=True,
    ax=axes,
    split_bars=split_bars,
)
plt.ylabel("count")
plt.xlabel("date of the first MRI (year)")
# plt.ylim(0, 22)
save_pub("date_before_therapy_MRI", bbox_inches="tight")
display.display(
    patient_data[patient_data.date_before_therapy_MRI_frac.notna()]
    .groupby("from_study")["date_before_therapy_MRI_frac"]
    .describe()
)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(WIDTH, 0.84 * WIDTH))
hatched_histplot(
    data=patient_data[patient_data.mri_1_to_op.notna()],
    hue="from_study",
    x="mri_1_to_op",
    multiple="layer",
    bins=np.arange(80, 181, 5),
    legend=True,
    ax=axes,
    split_bars=split_bars,
)
plt.ylabel("count")
plt.xlabel("time from MRI 1 to OP (days)")
plt.ylim(0, 22)
save_pub("mri_1_to_op", bbox_inches="tight")
display.display(
    patient_data[patient_data.mri_1_to_op.notna()]
    .groupby("from_study")["mri_1_to_op"]
    .describe()
)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(WIDTH, 0.84 * WIDTH))
hatched_histplot(
    data=patient_data[patient_data.mri_2_to_op.notna()],
    hue="from_study",
    x="mri_2_to_op",
    multiple="layer",
    bins=np.arange(0, 71, 4),
    legend=True,
    ax=axes,
    split_bars=split_bars,
)
plt.ylabel("count")
plt.xlabel("time from MRI 2 to OP (days)")
plt.ylim(0, 40)
save_pub("mri_2_to_op", bbox_inches="tight")
display.display(
    patient_data[patient_data.mri_2_to_op.notna()]
    .groupby("from_study")["mri_2_to_op"]
    .describe()
)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(WIDTH, 0.84 * WIDTH))
hatched_histplot(
    data=patient_data[patient_data.mri_1_to_mri_2.notna()],
    hue="from_study",
    x="mri_1_to_mri_2",
    multiple="layer",
    # bins=np.arange(0, 71, 4),
    legend=True,
    ax=axes,
    split_bars=split_bars,
)
plt.ylabel("count")
plt.xlabel("time from MRI 1 to MRI 2 (days)")
plt.ylim(0, 40)
save_pub("mri_1_to_mri_2", bbox_inches="tight")
display.display(
    patient_data[patient_data.mri_1_to_mri_2.notna()]
    .groupby("from_study")["mri_1_to_mri_2"]
    .describe()
)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(WIDTH, 0.84 * WIDTH))
hatched_histplot(
    data=patient_data[patient_data.mri_2_to_op.notna()],
    hue="location",
    x="mri_2_to_op",
    multiple="layer",
    bins=np.arange(10, 51, 4),
    hue_order=CENTER_ORDER,
    legend=True,
    ax=axes,
    split_bars=split_bars,
)
plt.ylabel("count")
plt.xlabel("time from MRI 2 to OP (days)")
# plt.ylim(0,40)
save_pub("mri_2_to_op", bbox_inches="tight")

# %%
# plot means and std
mean_stds_list = []
n_labels = []
modalities = ["T2w", "b800", "ADC"]
for lbl, data in tqdm(orig_dataset.items()):
    image_paths = [data_dir / p for p in data["images"]]
    images = [
        sitk.ReadImage(str(img_path)) for img_path in image_paths if img_path.exists()
    ]
    images_np = [sitk.GetArrayFromImage(img) for img in images]
    mean_stds_list += [
        {
            "Mean": np.mean(img[img > np.quantile(img, 0.01)]),
            "Std": np.std(img[img > np.quantile(img, 0.01)]),
            "Name": lbl,
            "Modality": mod,
        }
        for img, mod in zip(images_np, modalities)
    ]
    if "labels" in data:
        labels_image = sitk.ReadImage(str(data_dir / data["labels"]))
        labels_image_np = sitk.GetArrayFromImage(labels_image)
        n_labels.append((labels_image_np == 1).sum())

print(f"Average number of segmentation labels per image: {np.mean(n_labels):.0f}")
print(f"Median number of segmentation labels per image: {np.median(n_labels):.0f}")
labels_pp = np.sum(n_labels) / len(set(d.partition("_")[0] for d in orig_dataset))
print(f"Average number of segmentation labels per patient: {labels_pp:.0f}")

mean_stds = pd.DataFrame(mean_stds_list)
mean_stds["patientID"] = mean_stds["Name"].str.partition("_")[0]
mean_stds["Location"] = mean_stds["patientID"].apply(lambda s: s[:-3])
mean_stds = mean_stds.replace(
    {
        "1": "Center 1",
        "11": "Center 2",
        "99": "Center 3",
        "13": "Center 3",
        "12": "Center 4",
        "5": "Center 5",
        "8": "Center 6",
    }
)

# %%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6), sharey=True)
for num, (mod, ax_col) in enumerate(zip(modalities, axes.T)):
    data_mean_std = mean_stds.query(f"Modality == '{mod}' & Location != 'Center 6'")
    std_max = data_mean_std["Std"].quantile(0.99)
    data_mean_std = data_mean_std.drop(
        index=data_mean_std[data_mean_std.Std > std_max].index
    )
    sns.histplot(
        data=data_mean_std,
        x="Mean",
        hue="Location",
        stat="proportion",
        bins=20,
        common_norm=False,
        hue_order=CENTER_ORDER,
        ax=ax_col[0],
        legend=num == 0,
        element="poly",
    )

    sns.histplot(
        data=data_mean_std,
        x="Std",
        hue="Location",
        stat="proportion",
        bins=20,
        common_norm=False,
        hue_order=CENTER_ORDER,
        ax=ax_col[1],
        legend=False,
        element="poly",
    )

    ax_col[0].set_title(mod)

plt.tight_layout()
save_pub("mean-std-stat")

# %%

for mod, data in mean_stds[~mean_stds.Name.str.startswith("99")].groupby("Modality"):
    print(mod)
    display.display(data.sort_values("Mean").iloc[list(range(5)) + list(range(-5, 0))])
    display.display(data.sort_values("Std").iloc[list(range(5)) + list(range(-5, 0))])

# %%

image_names = [
    "11021_1_l0_d0",
    "1026_2_l0_d0",
    "1042_1_l0_d0",
    "1033_1_l0_d0",
    "13002_1_l1_d0",
    "13018_3_l0_d0",
]
display.display(acquisition_params.loc[image_names])
slices = [20, 20, 23, 17, 19, 25]
image_nums = np.arange(3).repeat(2)
vmax_list = [
    200,
    1250,
    15,
    200,
    1500,
    2500,
]

cbar_list = []
fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(6, 3))
for ax, name, slc, img_num, vmax in zip(
    axes.T.flat, image_names, slices, image_nums, vmax_list
):

    img_path = data_dir / orig_dataset[name]["images"][img_num]

    img = sitk.ReadImage(str(img_path))

    img_np = sitk.GetArrayFromImage(img)

    slice_img = img_np[slc]
    overlap = slice_img.shape[1] - slice_img.shape[0]
    assert overlap % 2 == 0
    if overlap != 0:
        slice_img = slice_img[:, overlap // 2 : -overlap // 2]

    if vmax is None:
        vmax = np.quantile(slice_img, 0.995)

    im = ax.imshow(
        slice_img,
        cmap="gray",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        extent=(0, slice_img.shape[1] / slice_img.shape[0], 0, 1),
    )
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, format="%4d")
    cbar.set_label("Intensity in a. u.")
    cbar_list.append(cbar)
    # print(f"{np.quantile(img_np, 0.995)},")

for ax, lbl in zip(axes[0], ["T2w", "b800", "ADC"]):
    ax.set_title(lbl, fontsize=10)

plt.tight_layout()

MAX_X = 9
for cbar_top, cbar_bottom in zip(cbar_list[::2], cbar_list[1::2]):
    y_lbl_top_pos = cbar_top.ax.yaxis.label.get_position()
    y_lbl_bottom_pos = cbar_bottom.ax.yaxis.label.get_position()

    cbar_top.ax.yaxis.set_label_coords(MAX_X, y_lbl_top_pos[1])
    cbar_bottom.ax.yaxis.set_label_coords(MAX_X, y_lbl_bottom_pos[1])

save_pub("example_images")

# %%

results_class.groupby(
    results_class["File Number"].str.partition("_")[0]
).dworak_ground_truth.mean().value_counts()

# %%
# count labelled voxels
mean_stds_list = []
n_labels = []
for lbl, data in tqdm(orig_dataset.items()):
    if "labels" in data:
        labels_image = sitk.ReadImage(str(data_dir / data["labels"]))
        labels_image_np = sitk.GetArrayFromImage(labels_image)
        n_labels.append((labels_image_np == 1).sum())

print(f"Total number of segmentation labels: {np.sum(n_labels):.0f}")
print(f"Average number of segmentation labels per image: {np.mean(n_labels):.0f}")
print(f"Median number of segmentation labels per image: {np.median(n_labels):.0f}")
labels_pp = np.sum(n_labels) / len(set(d.partition("_")[0] for d in orig_dataset))
print(f"Average number of segmentation labels per patient: {labels_pp:.0f}")
