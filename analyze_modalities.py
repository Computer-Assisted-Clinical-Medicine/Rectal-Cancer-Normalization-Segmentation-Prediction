"""
Analyze the results of the experiment
"""
# pylint:disable=duplicate-code

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

from utils import gather_results


def display_dataframe(dataframe: pd.DataFrame):
    display.display(display.HTML(dataframe.to_html()))


def display_markdown(text: str):
    display.display_markdown(display.Markdown(text))


def read_results(exp_dir: Path, versions=("best",)):
    """Read the results from Disk

    Parameters
    ----------
    exp_dir : Path
        The directory to read
    versions : tuple, optional
        The versions to analyze (final, best or both), by default ("best",)

    Returns
    -------
    pd.DataFrame
        The collected results
    """
    timepoints = pd.read_csv(
        Path(os.environ["data_dir"]) / "timepoints.csv", sep=";", index_col=0
    )

    collected_results = []
    for location in ["all", "Frankfurt", "Regensburg", "Mannheim-not-from-study"]:
        for external in [True, False]:
            for postprocessed in [True, False]:
                for version in versions:
                    loc_results = gather_results(
                        exp_dir / f"Normalization_{location}",
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

    res = pd.concat(collected_results)
    # reset index
    res.index = pd.RangeIndex(res.shape[0])
    # rename the File Number
    res = res.rename(columns={"File Number": "file_number"})
    # set timepoint
    res["timepoint"] = res.file_number.apply(lambda x: "_".join(x.split("_")[:2]))
    res["network"] = res.name.apply(lambda x: x.split("-")[0])
    # set treatment status
    mask = res.index[~res.timepoint.str.startswith("99")]
    res.loc[mask, "treatment_status"] = timepoints.loc[
        res.timepoint[mask]
    ].treatment_status.values
    mask = res.file_number.str.contains("_1_l") & res.timepoint.str.startswith("99")
    res.loc[mask, "treatment_status"] = "before therapy"
    mask = res.file_number.str.contains("_2_l") & res.timepoint.str.startswith("99")
    res.loc[mask, "treatment_status"] = "before OP"
    # set marker for before therapy
    res["before_therapy"] = res.treatment_status == "before therapy"

    def get_norm(name):
        if name == "combined_models":
            return "combined"
        else:
            return name.split("-")[-3]

    res["normalization"] = res.name.apply(get_norm)
    res["from_study"] = ~res.file_number.str.startswith("99")
    return res


# %%

display_markdown("## Load results")
base_path = Path("D:/Study Data/Experiments/Segmentation")
experiment_names = ["Normalization_Experiment", "T2_multicenter", "DWI_multicenter"]
training_modalities = ["both", "T2", "DWI"]

results_list = []
for exp_name, exp_mod in zip(experiment_names, training_modalities):
    exp_results = read_results(base_path / exp_name)
    exp_results["train_modality"] = exp_mod
    results_list.append(exp_results)
results = pd.concat(results_list)
# redo index
results.index = pd.RangeIndex(0, results.shape[0])

# drop all normalizations and versions not present for all experiments
results.drop(index=results.index[results.normalization == "MEAN_STD"], inplace=True)
results.drop(index=results.index[results.normalization == "HM_QUANTILE"], inplace=True)

display_markdown("## Load acquisition parameters")
with open(base_path / experiment_names[1] / "dataset.yaml") as f:
    orig_dataset = yaml.load(f, Loader=yaml.Loader)

new_root = Path("D:/Study Data/Dataset/Images")
# get image metadata
param_list = []
for number in results.file_number.unique():
    images = orig_dataset[number]["images"]
    param_file = new_root / images[0].parent.name / "acquisition_parameters.csv"
    parameters = pd.read_csv(param_file, sep=";", index_col=0)
    assert isinstance(parameters, pd.DataFrame)
    t2_params = parameters.loc[parameters.filename == images[0].name].copy()
    t2_params["file_number"] = number
    t2_params.set_index("file_number", inplace=True)
    t2_params["name"] = t2_params.index
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
results = results.merge(right=acquisition_params.location, on="file_number")

# set new names
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
results = results.replace(new_names)

# %% [markdown]

# ## Check that all files are there
# For each file, there should be:
# - 3 different modalities
# - 2 different networks
# - 2 different normalizations
# - 2 for postprocessed or not
#
# So 24 different in total, other differences are:
# - external
# - fold
# - train_location

# %%

n_images = (
    results.query("network != 'combined_models'")
    .groupby(["file_number", "external", "fold", "train_location"])
    .size()
)
incomplete = n_images[n_images != 24].index.to_frame().file_number.unique()
for num in incomplete:
    print(f"{num} is incomplete and will be deleted.")
    results.drop(index=results.index[results.file_number == num], inplace=True)

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
    hue="train_modality",
    row="normalization",
    kind="box",
    legend=True,
    legend_out=True,
)
g.fig.suptitle(
    "Modality, Overall Performance (version = best | before_therapy = True | postprocessed = True)"
)
g.fig.subplots_adjust(top=0.87)
plt.show()
plt.close()

g = sns.catplot(
    data=results.query(
        "version == 'best' & before_therapy & postprocessed & name != 'combined_models'"
    ),
    x="Dice",
    y="train_location",
    col="external",
    hue="normalization",
    row="train_modality",
    kind="box",
    legend=True,
    legend_out=True,
)
g.fig.suptitle(
    "Normalization, Overall Performance (version = best | before_therapy = True | postprocessed = True)"
)
g.fig.subplots_adjust(top=0.87)
plt.show()
plt.close()

g = sns.catplot(
    data=results.query("version == 'best' & before_therapy & postprocessed"),
    x="Dice",
    y="train_location",
    col="external",
    hue="network",
    row="train_modality",
    kind="box",
    legend=True,
    legend_out=True,
)
g.fig.suptitle(
    "Network, Overall Performance (version = best | before_therapy = True | postprocessed = True)"
)
g.fig.subplots_adjust(top=0.87)
plt.show()
plt.close()

# %%

data = results.query(
    "version == 'best' & before_therapy & postprocessed & external"
    + " & normalization == 'Perc' & network == 'UNet'"
)

# check that the Volume is consistent
for number, group_data in data.groupby("file_number"):
    close = np.allclose(
        group_data["Volume (L)"], group_data["Volume (L)"].iloc[0], atol=1, rtol=0.05
    )
    if not close:
        diff = group_data["Volume (L)"].max() - group_data["Volume (L)"].min()
        print(f"{number}: Maximum Volume Difference: {diff:.1f}")

data["Volume (L)"].hist()
plt.show()
plt.close()
data["Volume (L)"].hist(bins=np.arange(-5, 106, 10))
plt.show()
plt.close()

for mod, mod_data in data.groupby("train_modality"):
    volume = np.linspace(0, 90, num=20)
    spc = volume[1] - volume[0]
    dice = [
        mod_data.Dice[
            (mod_data["Volume (L)"] > v - spc / 2) & (mod_data["Volume (L)"] < v + spc / 2)
        ].mean()
        for v in volume
    ]
    dice_err = [
        mod_data.Dice[
            (mod_data["Volume (L)"] > v - spc / 2) & (mod_data["Volume (L)"] < v + spc / 2)
        ].std()
        for v in volume
    ]
    plt.scatter(volume, dice, label=mod)
    plt.plot(volume, dice, alpha=0.3)
plt.legend()
plt.xlabel("Label Volume in mm$^3$")
plt.ylabel("Mean Dice")
plt.title("Mean Dice depending on Label Volume (before therapy, external, Perc, UNet)")
plt.show()
plt.close()

for mod, mod_data in data.groupby("train_modality"):
    volume = np.linspace(0, 90, num=20)
    spc = volume[1] - volume[0]
    dice = [
        mod_data.Dice[
            (mod_data["Volume (P)"] > v - spc / 2) & (mod_data["Volume (P)"] < v + spc / 2)
        ].mean()
        for v in volume
    ]
    plt.scatter(volume, dice, label=mod)
    plt.plot(volume, dice, alpha=0.3)
plt.legend()
plt.xlabel("Predicted Volume in mm$^3$")
plt.ylabel("Mean Dice")
plt.title("Mean Dice depending on Predicted Volume (before therapy, external, Perc, UNet)")
plt.show()
plt.close()

for mod, mod_data in data.groupby("train_modality"):
    volume = np.linspace(0, 150)
    dice = [mod_data.Dice[mod_data["Volume (P)"] > v].mean() for v in volume]
    plt.plot(volume, dice, label=mod)
plt.legend()
plt.xlabel("Predicted Volume minimum Threshold in mm$^3$")
plt.ylabel("Mean Dice")
plt.title(
    "Mean Dice depending on min. Volume Threshold (before therapy, external, Perc, UNet)"
)
plt.show()
plt.close()

for mod, mod_data in data.groupby("train_modality"):
    volume = np.linspace(0, 150)
    dice = [mod_data.Dice[mod_data["Volume (L)"] > v].mean() for v in volume]
    plt.plot(volume, dice, label=mod)
plt.legend()
plt.xlabel("Label Volume minimum Threshold in mm$^3$")
plt.ylabel("Mean Dice")
plt.title(
    "Mean Dice depending on min. Volume Threshold (before therapy, external, Perc, UNet)"
)
plt.show()
plt.close()

for mod, mod_data in data.groupby("train_modality"):
    volume = np.linspace(0, 150)
    dice = [
        mod_data.Dice[(mod_data["Volume (L)"] > v) & (mod_data["Volume (P)"] > v)].mean()
        for v in volume
    ]
    plt.plot(volume, dice, label=mod)
plt.legend()
plt.xlabel("Label and Predicted Volume minimum Threshold in mm$^3$")
plt.ylabel("Mean Dice")
plt.title(
    "Mean Dice depending on min. Volume Threshold (before therapy, external, Perc, UNet)"
)
plt.show()
plt.close()

# %%
