# %% [markdown]
"""
# Analyze the trained network
## Import and Definitions
"""

import os
from pathlib import Path
import sys
import argparse
import matplotlib
import numpy as np
import seaborn as sns

from utils import gather_results

# if on cluster, use other backend
# pylint: disable=wrong-import-position, ungrouped-imports, wrong-import-order
if "CLUSTER" in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pylint: disable=pointless-string-statement


def init_argparse():
    """
    initialize the parser
    """
    argpar = argparse.ArgumentParser(description="Do the training analysis.")
    argpar.add_argument(
        "-p",
        "--path",
        metavar="path",
        nargs="?",
        type=str,
        help="The directory where all the experiments are.",
    )
    return argpar


# %% [markdown]
"""
## Set Paths
"""

parser = init_argparse()
args = parser.parse_args()

data_dir = Path(os.environ["data_dir"])
experiment_dir = Path(args.path)
plot_dir = experiment_dir / "analysis"

if not plot_dir.exists():
    plot_dir.mkdir()


def save_and_show(name: str):
    """Save the plot and if running in interactive mode, show the plot.

    Parameters
    ----------
    name : str
        The name to use when saving
    """
    # plt.savefig(plot_dir / f'{name}.pdf', dpi=600, facecolor='w')
    plt.savefig(plot_dir / f"{name}.png", dpi=600, facecolor="w")
    # only show in interactive mode
    if hasattr(sys, "ps1"):
        plt.show()
    plt.close()


def remove_parameters(name: str):
    # make nicer names
    for delete in ["DICE-Res-", "nBN-DO-", "-100"]:
        name = name.replace(delete, "")
    return name


def add_linebrakes(name: str, max_length=20):
    if len(name) > max_length and len(name) - max_length > max_length / 2:
        # add linebreak at the next -
        name = name[:max_length] + add_linebrakes(name[max_length:]).replace("-", "\n-", 1)
    return name


def remove_linebrakes(name: str):
    return name.replace("\n", "")


# %% [markdown]
"""
## Do the analysis for the training set
"""

VERSION = "best"

results = gather_results(experiment_dir, combined=True, version=VERSION)
# 1035_1 is with fat supression
results = results.drop(results.index[results["File Number"] == "1035_1"])
# make nicer names
results.name = results.name.cat.rename_categories(
    lambda x: add_linebrakes(remove_parameters(x))
)
results.sort_values("name", inplace=True)
# take the mean over all folds and models
results_mean = results.groupby("File Number").mean()
# take only the results from the study (without extra Mannheim data)
results_study_only = results.drop(
    results.index[results["File Number"].str.startswith("99")]
)

sns.catplot(data=results, y="name", x="Dice", kind="box", aspect=2)
save_and_show(f"{VERSION}_test_set_dice_models")

sns.catplot(data=results_study_only, y="name", x="Dice", kind="box", aspect=2)
save_and_show(f"{VERSION}_test_set_dice_models_study_only")

sns.catplot(data=results, y="name", x="IoU", kind="box", aspect=2)
save_and_show(f"{VERSION}_test_set_iou_models")

sns.catplot(data=results_study_only, y="name", x="IoU", kind="box", aspect=2)
save_and_show(f"{VERSION}_test_set_iou_models_study_only")

sns.catplot(data=results, y="name", x="Dice", hue="fold", kind="box", aspect=2)
save_and_show(f"{VERSION}_test_set_dice_models_folds")

sns.catplot(data=results_study_only, y="name", x="Dice", hue="fold", kind="box", aspect=2)
save_and_show(f"{VERSION}_test_set_dice_models_folds_study_only")

sns.catplot(data=results, y="fold", x="Dice", kind="box", aspect=2)
save_and_show(f"{VERSION}_test_set_dice_folds")

worst_files = list(results_mean.sort_values("Dice").iloc[:10].index)
worst_files_results = results[[f in worst_files for f in results["File Number"]]]
sns.catplot(
    data=worst_files_results, y="File Number", x="Dice", kind="box", order=worst_files
)
save_and_show(f"{VERSION}_test_set_dice_worst_files")

worst_files = list(results_mean.sort_values("Dice").iloc[:10].index)
worst_files_results = results[[f in worst_files for f in results["File Number"]]]
sns.catplot(
    data=worst_files_results, y="File Number", x="Dice", order=worst_files, hue="name"
)
save_and_show(f"{VERSION}_test_set_dice_worst_files_by_model")

plt.scatter(x=results_mean["Volume (L)"], y=results_mean["Dice"])
plt.xlabel("GT Volume")
plt.ylabel("Dice")
save_and_show(f"{VERSION}_test_set_volume_vs_dice")

plt.hist([results_mean["Volume (L)"].values, results_mean["Volume (P)"].values])
plt.xlabel("Volume")
plt.ylabel("Occurrence")
plt.legend(labels=["Ground Truth", "Predicted"])
save_and_show(f"{VERSION}_test_set_label_volume_hist")

plt.scatter(x=results_mean["Volume (L)"], y=results_mean["Hausdorff"])
plt.xlabel("GT Volume")
plt.ylabel("Hausdorff")
save_and_show(f"{VERSION}_test_set_volume_vs_hausdorff")

plt.scatter(x=results_mean["Volume (L)"], y=results_mean["Volume (P)"])
max_l = results_mean["Volume (L)"].max()
max_p = results_mean["Volume (P)"].max()
plt.plot([0, max_l], [0, max_p], color="gray")
plt.xlabel("GT Volume")
plt.ylabel("Predicted Volume")
save_and_show(f"{VERSION}_test_set_volume_vs_volume")

# remove linebreaks when exporting to csv
for res in [results, results_study_only]:
    res.name = results.name.cat.rename_categories(lambda x: remove_linebrakes(x))
results.groupby("name").describe().transpose().to_csv(
    plot_dir / f"{VERSION}_summary_test_set.csv", sep=";"
)
results_study_only.groupby("name").describe().transpose().to_csv(
    plot_dir / f"{VERSION}_summary_test_set_study_only.csv", sep=";"
)

# %% [markdown]
"""
## Do the analysis for the test-set
"""

results_ex = gather_results(experiment_dir, combined=True, external=True, version=VERSION)
if results_ex is None:
    print("No external testset")
    sys.exit()
results_ex = results_ex[np.logical_not(results_ex["File Number"].str.startswith("99"))]
# make nicer names
results_ex.name = results_ex.name.cat.rename_categories(
    lambda x: add_linebrakes(remove_parameters(x))
)
results_ex.sort_values("name", inplace=True)

sns.catplot(data=results_ex, y="name", x="Dice", kind="box", aspect=2)
save_and_show(f"{VERSION}_external_test_set_dice_models")

sns.catplot(data=results_ex, y="name", x="Dice", hue="fold", kind="box", aspect=2)
save_and_show(f"{VERSION}_external_test_set_dice_models_folds")

sns.catplot(data=results_ex, y="fold", x="Dice", kind="box", aspect=2)
save_and_show(f"{VERSION}_external_test_set_dice_folds")

sns.catplot(
    data=results_ex, y="File Number", x="Dice", hue="name", kind="box", aspect=1.4, height=6
)
save_and_show(f"{VERSION}_external_test_set_dice_files")

results_ex_mean = results_ex.groupby("File Number").mean()
plt.scatter(x=results_ex_mean["Volume (L)"], y=results_ex_mean["Dice"])
plt.xlabel("GT Volume")
plt.ylabel("Dice")
save_and_show(f"{VERSION}_external_test_set_volume_vs_dice")

plt.scatter(x=results_ex_mean["Volume (L)"], y=results_ex_mean["Volume (P)"])
max_l = results_ex_mean["Volume (L)"].max()
max_p = results_ex_mean["Volume (P)"].max()
plt.plot([0, max_l], [0, max_p], color="gray")
plt.xlabel("GT Volume")
plt.ylabel("Predicted Volume")
save_and_show(f"{VERSION}_external_test_set_volume_vs_volume")
