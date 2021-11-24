"""
Analyze the results of the experiment
"""

# %% [markdown]

# Analyze results
# # Import and load data

# %%

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from IPython import display
from scipy.stats import ttest_ind
from sklearn.cluster import DBSCAN, KMeans

from utils import gather_results


def display_dataframe(dataframe: pd.DataFrame):
    display.display(display.HTML(dataframe.to_html()))


def display_markdown(text: str):
    display.display_markdown(display.Markdown(text))


experiment_dir = Path(os.environ["experiment_dir"])

# load data
data_dir = Path(os.environ["data_dir"])
timepoints = pd.read_csv(data_dir / "timepoints.csv", sep=";", index_col=0)

with open(experiment_dir / "dataset.yaml") as f:
    orig_dataset = yaml.load(f, Loader=yaml.Loader)

collected_results = []
for location in ["all", "Frankfurt", "Regensburg", "Mannheim-not-from-study"]:
    print(f"{location}")
    for external in [True, False]:
        for postprocessed in [True, False]:
            for version in ["best", "final"]:
                loc_results = gather_results(
                    experiment_dir / f"Normalization_{location}",
                    external=external,
                    postprocessed=postprocessed,
                    version=version,
                )
                if loc_results is not None:
                    loc_results["train_location"] = location
                    loc_results["external"] = external
                    loc_results["postprocessed"] = postprocessed
                    loc_results["version"] = version
                    collected_results.append(loc_results)

results = pd.concat(collected_results)
# reset index
results.index = pd.RangeIndex(results.shape[0])
# set timepoint
results["timepoint"] = results["File Number"].apply(lambda x: "_".join(x.split("_")[:2]))
results["network"] = results.name.apply(lambda x: x.split("-")[0])
# set treatment status
mask = results.index[~results.timepoint.str.startswith("99")]
results.loc[mask, "treatment_status"] = timepoints.loc[
    results.timepoint[mask]
].treatment_status.values
mask = results["File Number"].str.contains("_1_l") & results.timepoint.str.startswith("99")
results.loc[mask, "treatment_status"] = "before therapy"
mask = results["File Number"].str.contains("_2_l") & results.timepoint.str.startswith("99")
results.loc[mask, "treatment_status"] = "before OP"
# set marker for before therapy
results["before_therapy"] = results.treatment_status == "before therapy"


def get_norm(name):
    if name == "combined_models":
        return "combined"
    else:
        return name.split("-")[-3]


results["normalization"] = results.name.apply(get_norm)
results["from_study"] = ~results["File Number"].str.startswith("99")

root = Path("D:/Study Data/Dataset/Images registered and N4 corrected")
new_root = Path("D:/Study Data/Dataset/Images")
# get image metadata
param_list = []
for number in results["File Number"].unique():
    images = orig_dataset[number]["images"]
    param_file = images[0].parent / "acquisition_parameters.csv"
    param_file = new_root / param_file.relative_to(root)
    parameters = pd.read_csv(param_file, sep=";", index_col=0)
    assert isinstance(parameters, pd.DataFrame)
    t2_params = parameters.loc[parameters.filename == images[0].name].copy()
    t2_params["File Number"] = number
    t2_params["name"] = t2_params.index
    t2_params.set_index("File Number", inplace=True)
    param_list.append(t2_params)
acquisition_params = pd.concat(param_list)
# drop columns that are mostly empty or always the same
for col in acquisition_params:
    num_na = acquisition_params[col].isna().sum()
    if num_na > acquisition_params.shape[0] // 2:
        acquisition_params.drop(columns=[col], inplace=True)
        continue
    same = (acquisition_params[col] == acquisition_params[col].iloc[0]).sum()
    if same > acquisition_params.shape[0] * 0.9:
        acquisition_params.drop(columns=[col], inplace=True)
        continue
# correct pixel spacing
def func(x):
    if "\\" in x:
        return float(x.split("\\")[0])
    else:
        return float(x[1:].split(",")[0])


acquisition_params.pixel_spacing = acquisition_params["0028|0030"].apply(func)
# correct location
acquisition_params.loc[
    acquisition_params.index.str.startswith("99"), "location"
] = "Mannheim-not-from-study"

# filter important parameters
column_names = {
    "0008|1090": "model_name",
    "0018|0050": "slice_thickness",
    "0018|0080": "repetition_time",
    "0018|0095": "pixel_bandwidth",
    "0018|1314": "flip_angle",
    "0018|0081": "echo_time",
    "0018|0087": "field_strength",
    "pixel_spacing": "pixel_spacing",
    "location": "location",
}
acquisition_params = acquisition_params[column_names.keys()]
acquisition_params.rename(columns=column_names, inplace=True)

# set location
results = results.merge(right=acquisition_params.location, on="File Number")

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

g = sns.catplot(
    data=results.query(
        "version == 'best' & before_therapy & postprocessed & name != 'combined_models'"
    ),
    x="Dice",
    y="train_location",
    col="external",
    hue="normalization",
    row="network",
    kind="box",
    legend=True,
    legend_out=True,
)
g.fig.suptitle(
    "Overall Performance (version = best | before_therapy = True | postprocessed = True)"
)
g.fig.subplots_adjust(top=0.87)
plt.show()
plt.close()

g = sns.catplot(
    data=results.query(
        "version == 'best' & before_therapy & postprocessed & name == 'combined_models'"
    ),
    x="Dice",
    y="train_location",
    col="external",
    hue="normalization",
    row="network",
    kind="box",
    legend=True,
    legend_out=True,
)
g.fig.suptitle(
    "Combined networks (version = best | before_therapy = True | postprocessed = True)"
)
g.fig.subplots_adjust(top=0.88)
plt.show()
plt.close()

g = sns.catplot(
    data=results.query(
        "version == 'best' & before_therapy & postprocessed & name != 'combined_models' & train_location != 'all'"
    ),
    x="Dice",
    y="normalization",
    col="external",
    hue="network",
    kind="box",
    legend=True,
    legend_out=True,
)
g.fig.suptitle(
    "Performance on all locations (except all) (version = best | before_therapy = True | postprocessed = True)"
)
g.fig.subplots_adjust(top=0.87)
plt.show()
plt.close()

display_markdown("All training locations except all.")
display_dataframe(
    pd.DataFrame(
        results.query(
            "version == 'best' & before_therapy & postprocessed"
            + " & name != 'combined_models' & train_location != 'all'"
        )
        .groupby(["normalization", "external"])
        .Dice.median()
    )
)

display_markdown("All training locations except all.")
display_dataframe(
    pd.DataFrame(
        results.query(
            "version == 'best' & before_therapy & postprocessed"
            + " & name != 'combined_models' & train_location != 'all'"
        )
        .groupby(["normalization", "network", "external"])
        .Dice.median()
    )
)

display_markdown("Models trained on all images.")
display_dataframe(
    pd.DataFrame(
        results.query(
            "version == 'best' & before_therapy & postprocessed"
            + " & name != 'combined_models' & train_location == 'all'"
        )
        .groupby(["normalization", "network"])
        .Dice.median()
    )
)

display_markdown("Model Comparison (internal)")
display_dataframe(
    pd.DataFrame(
        results.query(
            "version == 'best' & before_therapy & postprocessed" + " & not external"
        )
        .groupby(["train_location", "normalization", "network"])
        .Dice.median()
    )
)
display_markdown("Model Comparison (external)")
display_dataframe(
    pd.DataFrame(
        results.query("version == 'best' & before_therapy & postprocessed & external")
        .groupby(["train_location", "normalization", "network"])
        .Dice.median()
    )
)

# %%

for train_location in results.train_location.unique():
    g = sns.catplot(
        data=results.query(
            f"version == 'best' & train_location == '{train_location}'"
            + " & before_therapy & name != 'combined_models'"
        ),
        x="Dice",
        y="normalization",
        col="postprocessed",
        hue="external",
        row="network",
        kind="box",
        legend=True,
        legend_out=True,
    )
    g.fig.suptitle(f"Location = {train_location} (version = best | before_therapy = True)")
    g.fig.subplots_adjust(top=0.88)
    plt.show()
    plt.close()

    g = sns.catplot(
        data=results.query(
            f"version == 'best' & train_location == '{train_location}'"
            + " & name != 'combined_models' & external"
        ),
        x="Dice",
        y="normalization",
        col="postprocessed",
        hue="before_therapy",
        row="network",
        kind="box",
        legend=True,
        legend_out=True,
    )
    g.fig.suptitle(f"Location = {train_location} (version = best | external = True)")
    g.fig.subplots_adjust(top=0.88)
    plt.show()
    plt.close()

    g = sns.catplot(
        data=results.query(
            f"version == 'best' & train_location == '{train_location}'"
            + " & name != 'combined_models' & external & postprocessed"
        ),
        x="Dice",
        y="normalization",
        col="before_therapy",
        hue="from_study",
        row="network",
        kind="box",
        legend=True,
        legend_out=True,
    )
    g.fig.suptitle(
        f"Location = {train_location} (version = best | external = True | postprocessed = True)"
    )
    g.fig.subplots_adjust(top=0.88)
    plt.show()
    plt.close()

    g = sns.catplot(
        data=results.query(
            f"version == 'best' & train_location == '{train_location}'"
            + " & name != 'combined_models' & not external & postprocessed"
        ),
        x="Dice",
        y="normalization",
        col="before_therapy",
        hue="from_study",
        row="network",
        kind="box",
        legend=True,
        legend_out=True,
    )
    g.fig.suptitle(
        f"Location = {train_location} (version = best | external = False | postprocessed = True)"
    )
    g.fig.subplots_adjust(top=0.88)
    plt.show()
    plt.close()

# %%

images_dice = pd.DataFrame(index=results["File Number"].unique())
images_dice["minimum"] = results.groupby("File Number").Dice.min()
images_dice["mean"] = results.groupby("File Number").Dice.mean()
images_dice["median"] = results.groupby("File Number").Dice.median()
images_dice["maximum"] = results.groupby("File Number").Dice.max()

display_markdown("## Bad images\n### Mean")
display_dataframe(images_dice.sort_values("mean").iloc[:20])

display_markdown("### Median")
display_dataframe(images_dice.sort_values("median").iloc[:20])

display_markdown("### Maximum")
display_dataframe(images_dice.sort_values("maximum").iloc[:20])

# %%

display_markdown("## Analyze the different centers")

# remove Regensburg UK, there is just one value
to_drop = acquisition_params.index[acquisition_params.location == "Regensburg-UK"]

g = sns.displot(
    data=acquisition_params.drop(to_drop),
    hue="model_name",
    y="location",
    multiple="stack",
    edgecolor="grey",
    linewidth=1.5,
)
plt.show()
plt.close()

g = sns.displot(
    data=acquisition_params.drop(to_drop),
    hue="location",
    x="slice_thickness",
    binrange=(0.5, 6.5),
    binwidth=1,
    multiple="stack",
)
plt.show()
plt.close()

g = sns.displot(
    data=acquisition_params.drop(to_drop),
    hue="location",
    x="repetition_time",
    multiple="layer",
    element="step",
    kde=True,
)
plt.show()
plt.close()

g = sns.displot(
    data=acquisition_params.drop(to_drop),
    hue="location",
    x="echo_time",
    multiple="stack",
    binwidth=10,
    binrange=(75, 135),
)
plt.show()
plt.close()

g = sns.displot(
    data=acquisition_params.drop(to_drop),
    hue="location",
    x="pixel_bandwidth",
    multiple="stack",
    binwidth=50,
)
plt.show()
plt.close()

g = sns.displot(
    data=acquisition_params.drop(to_drop),
    hue="location",
    x="flip_angle",
    multiple="stack",
    binwidth=10,
    binrange=(85, 165),
)
plt.show()
plt.close()

g = sns.displot(
    data=acquisition_params.drop(to_drop),
    hue="location",
    x="field_strength",
    multiple="dodge",
    binwidth=1.5,
    binrange=(0.75, 3.75),
)
g.facet_axis(0, 0).set_xticks([1.5, 3])
plt.show()
plt.close()

g = sns.displot(
    data=acquisition_params.drop(to_drop),
    hue="location",
    x="pixel_spacing",
    multiple="layer",
    binwidth=0.2,
    binrange=(0.1, 1.7),
    element="step",
    kde=True,
)
plt.show()
plt.close()

# %%

display_markdown("## Look for correlation\n### Group by training location")

results_merged = results.merge(
    acquisition_params.drop(columns="location"), on="File Number"
)

for loc in results.train_location.unique():
    g = sns.histplot(
        data=results_merged.query(
            f"train_location == '{loc}' & normalization == 'QUANTILE'"
            + " & before_therapy & location != 'Regensburg-UK'"
        ),
        x="Dice",
        hue="location",
        kde=True,
        multiple="layer",
        element="step",
        stat="percent",
        common_norm=False,
        binwidth=0.05,
    )
    g.set_title(f"Training location = {loc}")

    plt.show()
    plt.close()

display_markdown("### Group by location")

for loc in results.location.unique():
    if loc == "Regensburg-UK":
        continue
    g = sns.histplot(
        data=results_merged.query(
            "normalization == 'QUANTILE'" + f" & before_therapy & location == '{loc}'"
        ),
        x="Dice",
        hue="train_location",
        kde=True,
        multiple="layer",
        element="step",
        stat="percent",
        common_norm=False,
        binwidth=0.05,
    )
    g.set_title(f"location = {loc}")

    plt.show()
    plt.close()

display_markdown("### Try to find correlations")
# %%
for loc in results.train_location.unique():
    for col in acquisition_params.columns:
        data = results_merged.query(
            f"train_location == '{loc}' & normalization == 'QUANTILE'"
            + " & before_therapy & location != 'Regensburg-UK'"
        )
        if not pd.api.types.is_numeric_dtype(results_merged[col].dtype):
            g = sns.histplot(
                data=data,
                x="Dice",
                hue=col,
                kde=True,
                common_norm=False,
                stat="percent",
                multiple="layer",
            )
            g.set_title(f"train_location = {loc} : {col}")
        plt.show()
        plt.close()
# %%
for col in acquisition_params.columns:
    if not pd.api.types.is_numeric_dtype(results_merged[col].dtype):
        continue
    data = results_merged.query("normalization == 'QUANTILE' & before_therapy").copy()

    N_BINS = 10

    g = sns.lmplot(
        data=data,
        y="Dice",
        x=col,
        markers=["."] * data.train_location.nunique(),
        hue="train_location",
        truncate=True,
        col="external",
    )
    g.facet_axis(0, 0).set_title(f"{col}")
    plt.show()
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(8, 4))
    for external, ax in zip([False, True], axes):
        plot_data = data[data.external == external].copy()
        bins = np.linspace(
            start=plot_data[col].min() - 0.01,
            stop=plot_data[col].max() + 0.01,
            num=N_BINS + 1,
        )
        bin_num = pd.Series(
            np.digitize(plot_data[col], bins),
            index=plot_data.index,
            name="bin_num",
        )
        plot_data[col + "_bin"] = bin_num
        bin_centers = bins[:-1] + (bins[1] - bins[0]) / 2
        plot_data[col + "_bin_center"] = bin_centers[plot_data[col + "_bin"] - 1]

        groups = plot_data.Dice.groupby([plot_data.train_location, bin_num])
        means = groups.mean()
        stds = groups.std()

        for loc in plot_data.train_location.unique():
            ax.errorbar(
                x=bin_centers[means.loc[loc].index - 1],
                y=means.loc[loc],
                yerr=stds.loc[loc],
                label=loc,
                capsize=5,
                capthick=2,
            )
        ax.set_title(f"external = {external}")
        ax.set_xlabel(f"{col}")

    axes[1].legend()
    axes[0].set_ylabel("Dice")
    fig.tight_layout()

    plt.show()
    plt.close()

# %%
for col in acquisition_params.columns:
    if not pd.api.types.is_numeric_dtype(results_merged[col].dtype):
        continue
    data = results_merged.query("before_therapy & normalization != 'combined'").copy()

    g = sns.lmplot(
        data=data,
        y="Dice",
        x=col,
        markers=["."] * 4,
        hue="normalization",
        truncate=True,
        col="external",
    )
    plt.show()
    plt.close()

    bins = np.linspace(
        start=data[col].min() - 0.01,
        stop=data[col].max() + 0.01,
        num=21,
    )
    bin_num = pd.Series(
        np.digitize(data[col], bins),
        index=data.index,
        name="bin_num",
    )
    data[col + "_bin"] = bin_num
    bin_centers = bins[:-1] + (bins[1] - bins[0]) / 2
    data[col + "_bin_center"] = bin_centers[data[col + "_bin"] - 1]

    groups = data.Dice.groupby([data.normalization, bin_num])
    means = groups.mean()
    stds = groups.std()

    plt.figure(figsize=(8, 6))
    for loc in data.normalization.unique():
        plt.errorbar(
            x=bin_centers[means.loc[loc].index - 1],
            y=means.loc[loc],
            yerr=stds.loc[loc],
            label=loc,
            capsize=5,
            capthick=2,
        )
    plt.legend()
    plt.show()
    plt.close()

    for df in [means, stds]:
        df = df.reset_index()
        df[col] = bin_centers[df.bin_num - 1]

    g = sns.lineplot(
        data=data,
        x=col + "_bin_center",
        y="Dice",
        hue="normalization",
        markers=True,
        err_style="bars",
    )
    g.set_xlabel(col)
    for line in g.lines:
        line.set_alpha(0)
        line.set_marker("o")
    plt.show()
    plt.close()

# %%

display_markdown("## Publication figures")

new_names = {
    "QUANTILE": "Perc",
    "HM_QUANTILE": "Perc-HM",
    "MEAN_STD": "M-Std",
    "HISTOGRAM_MATCHING": "HM",
    "UNet2D": "UNet",
    "DeepLabv3plus2D": "DeepLabV3+",
    "Frankfurt": "Center 1",
    "Regensburg": "Center 2",
    "Mannheim-not-from-study": "Center 3",
    "Mannheim": "Center 4",
    "Wuerzburg": "Center 5",
    "Regensburg-UK": "Center 6",
}
norm_order = ["Perc", "Perc-HM", "HM", "M-Std"]

TITLE = "Performance when training on a single center"
display_markdown(TITLE)
data = results.query(
    "version == 'best' & before_therapy & postprocessed & name != 'combined_models'"
).replace(new_names)
data.loc[data.train_location == "all", "external"] = "test"

g = sns.catplot(
    data=data,
    y="Dice",
    x="normalization",
    order=norm_order,
    col="external",
    hue="network",
    kind="box",
    legend=True,
    legend_out=True,
    height=4,
    aspect=0.7,
)
titles = [
    "A",
    "B",
    "C",
]
for ax, tlt in zip(g.axes[0], titles):
    ax.set_title(tlt)
plt.show()
plt.close()

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(8, 4))

sns.histplot(
    data=acquisition_params.replace(new_names),
    hue="location",
    x="pixel_spacing",
    multiple="stack",
    binwidth=0.2,
    binrange=(0, 1.8),
    hue_order=[f"Center {i}" for i in range(1, 7)],
    ax=axes[0],
)
axes[0].set_xlabel("in-plane resolution (mm)")

sns.histplot(
    data=acquisition_params.replace(new_names),
    hue="location",
    x="echo_time",
    multiple="stack",
    binwidth=10,
    binrange=(75, 135),
    hue_order=[f"Center {i}" for i in range(1, 7)],
    ax=axes[1],
    legend=False,
)
axes[1].set_xlabel("echo time (ms)")

titles = [
    "A",
    "B",
]
for ax, tlt in zip(axes, titles):
    ax.set_title(tlt)

plt.tight_layout()
plt.show()
plt.close()

raw_data = (
    results.query(
        "version == 'best' & before_therapy & postprocessed"
        + " & name != 'combined_models'"
    )
    .replace(new_names)
    .groupby(["normalization", "network"])
)
data = pd.DataFrame(raw_data.Dice.mean()).drop("Dice", axis="columns")
for group in raw_data.groups:
    group_data = raw_data.get_group(group)
    mean_all = group_data[group_data.train_location == "all"].Dice.mean()
    data.loc[group, "all"] = mean_all
    mean_internal = group_data[
        (group_data.train_location != "all") & (~group_data.external)
    ].Dice.mean()
    data.loc[group, "internal"] = mean_internal
    mean_external = group_data[
        (group_data.train_location != "all") & group_data.external
    ].Dice.mean()
    data.loc[group, "external"] = mean_external
display_markdown("Models trained on one center.")
display_dataframe(data.round(2))

# %%
display_markdown("## Compare networks")

custom_names = {**new_names}
custom_names["DeepLabv3plus2D"] = "DLv3+"

raw_data = (
    results.query(
        "version == 'best' & before_therapy & postprocessed"
        + " & name != 'combined_models' & train_location == 'all'"
    )
    .replace(custom_names)
    .groupby(["normalization", "network"])
    .Dice
)
data = pd.DataFrame(raw_data.median())

display_markdown("Models trained on all images.")
display_dataframe(data.round(2))
differences = pd.DataFrame(index=data.index, columns=data.index)

for first in data.index:
    for second in data.index:
        tscore, pvalue = ttest_ind(raw_data.get_group(first), raw_data.get_group(second))
        differences.loc[first, second] = np.round(pvalue, 3)
display_markdown("P-Values")
display_dataframe(differences.round(2))
display_dataframe(differences < 0.05)

display_markdown("## Compare Normalization")

display_markdown("### all")
raw_data = (
    results.query(
        "version == 'best' & before_therapy & postprocessed"
        + " & name != 'combined_models' & train_location == 'all' & not external"
    )
    .replace(custom_names)
    .groupby(["normalization", "network"])
    .Dice
)
data = pd.DataFrame(raw_data.mean())
differences = pd.DataFrame(index=data.index, columns=data.index)
for first in data.index:
    for second in data.index:
        tscore, pvalue = ttest_ind(raw_data.get_group(first), raw_data.get_group(second))
        differences.loc[first, second] = np.round(pvalue, 3)
display_dataframe(data.round(2))
display_markdown("P-Values")
display_dataframe(differences)
display_dataframe(differences < 0.05)

display_markdown("### internal")
raw_data = (
    results.query(
        "version == 'best' & before_therapy & postprocessed"
        + " & name != 'combined_models' & train_location != 'all' & not external"
    )
    .replace(custom_names)
    .groupby(["normalization", "network"])
    .Dice
)
data = pd.DataFrame(raw_data.mean())
differences = pd.DataFrame(index=data.index, columns=data.index)
for first in data.index:
    for second in data.index:
        tscore, pvalue = ttest_ind(raw_data.get_group(first), raw_data.get_group(second))
        differences.loc[first, second] = np.round(pvalue, 3)
display_dataframe(data.round(2))
display_markdown("P-Values")
display_dataframe(differences)
display_dataframe(differences < 0.05)

display_markdown("### external")
raw_data = (
    results.query(
        "version == 'best' & before_therapy & postprocessed"
        + " & name != 'combined_models' & train_location != 'all' & external"
    )
    .replace(custom_names)
    .groupby(["normalization", "network"])
    .Dice
)
data = pd.DataFrame(raw_data.mean())
differences = pd.DataFrame(index=data.index, columns=data.index)
for first in data.index:
    for second in data.index:
        tscore, pvalue = ttest_ind(raw_data.get_group(first), raw_data.get_group(second))
        differences.loc[first, second] = np.round(pvalue, 3)
display_dataframe(data.round(2))
display_markdown("P-Values")
display_dataframe(differences)
display_dataframe(differences < 0.05)
display_dataframe(differences < 0.01)

# %%

display_markdown("# Cluster images")

to_cluster = acquisition_params.drop(["model_name", "location"], axis="columns")
means = to_cluster.apply(pd.Series.mean)
stds = to_cluster.apply(pd.Series.std)
to_cluster_norm = (to_cluster - means) / stds
clustered = acquisition_params.copy()

kmeans = KMeans(n_clusters=10, random_state=0).fit(to_cluster_norm)
clustered["KMeans"] = kmeans.labels_
dbscan = DBSCAN(eps=0.3, min_samples=2, metric="cosine").fit(to_cluster_norm)
clustered["DBSCAN"] = dbscan.labels_

for col in to_cluster_norm:
    g = sns.histplot(
        x=clustered[col],
        hue=clustered["KMeans"].astype(str),
        hue_order=np.sort(clustered["KMeans"].unique()).astype(str),
        multiple="stack",
        kde=True,
        stat="percent",
    )
    g.set_title(f"{col} : K-Means")
    plt.show()
    plt.close()

    g = sns.histplot(
        x=clustered[col],
        hue=clustered["DBSCAN"].astype(str),
        hue_order=np.sort(clustered["DBSCAN"].unique()).astype(str),
        multiple="stack",
        kde=True,
        stat="percent",
    )
    g.set_title(f"{col} : DBSCAN")
    plt.show()
    plt.close()
