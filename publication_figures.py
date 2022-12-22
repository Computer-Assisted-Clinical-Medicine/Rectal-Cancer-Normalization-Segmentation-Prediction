"""Make the figures for the paper"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import SimpleITK as sitk
import yaml
from matplotlib.transforms import Bbox
from tqdm import tqdm

from plot_utils import create_axes, display_dataframe, display_markdown, plot_significance
from SegClassRegBasis import normalization
from utils import calculate_auc, gather_all_results


def save_pub(filename, **kwargs):
    """Save the figure for publication"""
    pub_dir = Path(os.environ["experiment_dir"]) / "Normalization_Experiment" / "pub"
    if not pub_dir.exists():
        pub_dir.mkdir()
    plt.savefig(pub_dir / f"{filename}.png", dpi=600, **kwargs)
    # plt.savefig(pub_dir / f"{name}.pdf", dpi=600, **kwargs)
    plt.show()
    plt.close()


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

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
classification_tasks = [
    c.partition("_accuracy")[0] for c in results_class if "accuracy" in c
]

results_reg, _ = gather_all_results(task="regression")
regression_tasks = [c.partition("_rmse")[0] for c in results_reg if c.endswith("_rmse")]

# set the experiment type
for df in [results_seg, results_class, results_reg]:
    df.loc[df.train_location.apply(lambda x: "Not" in x), "experiment_type"] = "except_one"
    df.loc[
        df.train_location.apply(lambda x: "Not" not in x), "experiment_type"
    ] = "single_center"
    df.loc[df.train_location == "all", "experiment_type"] = "all"

# %%

print("Calculate AUC values")

res_list = []
for class_task in classification_tasks:
    # calculate AUC values
    res_tsk = results_class.groupby(
        [
            "experiment_type",
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

new_names = {
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
    "Mannheim-not-from-study": "Center 3",
    "Mannheim": "Center 4",
    "Wuerzburg": "Center 5",
    "Regensburg-UK": "Center 6",
    "Not-Frankfurt": "Not Center 1",
    "Not-Regensburg": "Not Center 2",
    "Not-Mannheim": "Not Center 3 and 4",
}
norm_order = [
    "GAN-Def",
    "GAN-Seg",
    "GAN-Img",
    "GAN-Win",
    "GAN-No-ed",
    "Perc",
    "Perc-HM",
    "HM",
    "M-Std",
    "Win",
]
external_order = ["test", False, True]

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
    df.replace(new_names, inplace=True)
    df.normalization = df.normalization.cat.remove_categories(
        [c for c in df.normalization.cat.categories if c not in norm_order]
    )

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


def turn_ticks(axes_to_turn):
    for label in axes_to_turn.get_xticklabels():
        label.set_ha("right")
        label.set_rotation(40)


for experiment_type in ["all", "single_center", "except_one"]:

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
        ax=axes[1, 2],
    )
    axes[1, 2].set_title("Age")
    axes[1, 2].set_ylabel("RMSE")
    axes[1, 2].set_ylim(-1, 25)
    turn_ticks(axes[1, 2])

    plt.tight_layout()
    save_pub(f"{experiment_type}_summary")
    plt.show()
    plt.close()

# %%

print("Performance in all experiments")

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20 / 2.54, 16 / 2.54))

data_seg = results_seg_new_names.query(
    "before_therapy & postprocessed & name != 'combined_models' & version == 'best'"
)
data_class = results_class_task_new_names.query("version == 'best'")
data_reg = results_reg_new_names.query("version == 'best'")

sns.boxplot(
    data=data_seg.query("do_batch_normalization"),
    x="normalization",
    y="Dice",
    hue="experiment_type",
    ax=axes[0, 0],
)
axes[0, 0].set_title("Segmentation Batch Norm")
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
        ax=ax,
    )
    ax.set_title(task.replace("_", " ").capitalize())
    ax.set_ylabel("AUC")
    turn_ticks(ax)
    if task != "sex":
        ax.legend([], [], frameon=False)

axes[0, 2].legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0,
    title="Experiment Type",
)
new_labels = ["all", "except one", "single center"]
for t, l in zip(axes[0, 2].get_legend().texts, new_labels):
    t.set_text(l)
axes[1, 0].set_ylim(0.295, 0.755)
axes[1, 1].set_ylim(0.295, 0.805)

sns.boxplot(
    data=data_reg,
    x="normalization",
    y="age_rmse",
    hue="experiment_type",
    ax=axes[1, 2],
)
axes[1, 2].set_title("Age")
axes[1, 2].set_ylabel("RMSE")
turn_ticks(axes[1, 2])
axes[1, 2].set_ylim(-1, 25)
axes[1, 2].legend([], [], frameon=False)

plt.tight_layout()
save_pub("all_experiment_types_summary")
plt.show()
plt.close()

# %%

print("Look for significance")

for experiment_type in ["all", "single_center", "except_one"]:
    display_markdown(f"### {experiment_type}")

    data_seg = results_seg_new_names.query(
        "before_therapy & postprocessed & name != 'combined_models' & version == 'best'"
        f" & experiment_type == '{experiment_type}'"
    )
    if experiment_type == "all":
        grouped_data = data_seg.groupby(["do_batch_normalization", "normalization"])
        plot_significance(grouped_data, f"{experiment_type} - Segmentation", "Dice")
    else:
        for ext in ["not external", "external"]:
            data_seg_ext = data_seg.query(ext)
            grouped_data = data_seg_ext.groupby(["do_batch_normalization", "normalization"])
            plot_significance(
                grouped_data, f"{experiment_type} - {ext} - Segmentation", "Dice"
            )

    for task in good_classification_tasks:
        data_class = results_class_task_new_names.query(
            f"experiment_type == '{experiment_type}' & version == 'best' & task == '{task}'"
        )
        # only use first label to not overstate the significance
        first_label = data_class.label.unique()[0]
        data_class = data_class.query(f"label == '{first_label}'")
        if experiment_type == "all":
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
    if experiment_type == "all":
        grouped_data = data_reg.groupby(["normalization"])
        plot_significance(grouped_data, f"{experiment_type} - Age", "age_rmse")
    else:
        for ext in ["not external", "external"]:
            grouped_data = data_reg.query(ext).groupby(["normalization"])
            plot_significance(grouped_data, f"{experiment_type} - {ext} - Age", "age_rmse")


# %%

display_markdown("### Calculations for the paper")

seg_all = results_seg_new_names.query(
    "before_therapy & postprocessed & name != 'combined_models' & version == 'best'"
    " & experiment_type == 'all'"
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
# make a nice description of the hm methods
images = orig_dataset["1001_1_l0_d0"]["images"]
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
        img_np[img_np.shape[0] // 2], interpolation="nearest", cmap="gray", vmin=-1, vmax=1
    )
    ax_line[2].axis("off")
    if i == 0:
        ax_line[2].set_title("Original Image")

    ax_line[3].imshow(
        img_norm_np[img_norm_np.shape[0] // 2],
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
images = orig_dataset["1001_1_l0_d0"]["images"]
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
        img_np[img_np.shape[0] // 2],
        interpolation="nearest",
        cmap="gray",
        vmin=img_np.min(),
        vmax=img_np.max(),
    )
    ax_line[2].axis("off")
    if i == 0:
        ax_line[2].set_title("Original Image")

    ax_line[3].imshow(
        img_norm_np[img_norm_np.shape[0] // 2],
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
images = orig_dataset["1001_1_l0_d0"]["images"]
images_sitk = [sitk.ReadImage(str(data_dir / img)) for img in images]
norm_dir_hist = experiment_dir / "data_preprocessed" / "MEAN_STD"
norm_files = [norm_dir_hist / f"normalization_mod{i}.yaml" for i in range(3)]
norms = [normalization.MeanSTD.from_file(f) for f in norm_files]
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
        img_np[img_np.shape[0] // 2],
        interpolation="nearest",
        cmap="gray",
        vmin=img_np.min(),
        vmax=img_np.max(),
    )
    ax_line[2].axis("off")
    if i == 0:
        ax_line[2].set_title("Original Image")

    ax_line[3].imshow(
        img_norm_np[img_norm_np.shape[0] // 2],
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
images = orig_dataset["1001_1_l0_d0"]["images"]
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
        img_np[img_np.shape[0] // 2],
        interpolation="nearest",
        cmap="gray",
        vmin=img_np.min(),
        vmax=img_np.max(),
    )
    ax_line[2].axis("off")
    if i == 0:
        ax_line[2].set_title("Original Image")

    ax_line[3].imshow(
        img_norm_np[img_norm_np.shape[0] // 2],
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
images = orig_dataset["1001_1_l0_d0"]["images"]
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
        img_np[img_np.shape[0] // 2], interpolation="nearest", cmap="gray", vmin=-1, vmax=1
    )
    ax_line[2].axis("off")
    if i == 0:
        ax_line[2].set_title("Original Image")

    ax_line[3].imshow(
        img_norm_np[img_norm_np.shape[0] // 2],
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
# plot means and std
mean_stds_list = []
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

mean_stds = pd.DataFrame(mean_stds_list)
mean_stds["patientID"] = mean_stds["Name"].str.partition("_")[0]
mean_stds["Location"] = mean_stds["patientID"].apply(lambda s: s[:-3])
mean_stds = mean_stds.replace(
    {
        "1": "Center 1",
        "11": "Center 2",
        "99": "Center 3",
        "13": "Center 4",
        "12": "Center 5",
        "5": "Center 6",
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
        hue_order=[f"Center {i}" for i in range(1, 6)],
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
        hue_order=[f"Center {i}" for i in range(1, 6)],
        ax=ax_col[1],
        legend=False,
        element="poly",
    )

    ax_col[0].set_title(mod)

plt.tight_layout()
save_pub("mean-std-stat")
