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
from IPython import display

from utils import gather_results


def display_dataframe(dataframe: pd.DataFrame):
    display.display(display.HTML(dataframe.to_html()))


experiment_dir = Path(os.environ["experiment_dir"])

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
# set marker for before therapy
results["before_therapy"] = np.logical_not(results["File Number"].str.contains("_2_l"))
results["network"] = results.name.apply(lambda x: x.split("-")[0])


def get_norm(name):
    if name == "combined_models":
        return "combined"
    else:
        return name.split("-")[-3]


results["normalization"] = results.name.apply(get_norm)
results["from_study"] = np.logical_not(results["File Number"].str.startswith("99"))

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

display_dataframe(
    pd.DataFrame(
        results.query(
            "version == 'best' & before_therapy & postprocessed"
            + " & name != 'combined_models' & train_location == 'all'"
        )
        .groupby(["normalization", "external"])
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
