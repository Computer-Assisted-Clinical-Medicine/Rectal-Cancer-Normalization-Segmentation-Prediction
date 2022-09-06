"""Check the normalization to see if there are any large deviations"""
# %%

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from IPython.display import display
from tqdm.autonotebook import tqdm

from SegClassRegBasis import evaluation

# %%

data_dir = Path(os.environ["data_dir"])
experiment_dir = Path(os.environ["experiment_dir"])
GROUP_BASE_NAME = "Normalization_Experiment"
exp_group_base_dir = experiment_dir / GROUP_BASE_NAME

quantile_location = exp_group_base_dir / "data_preprocessed" / "QUANTILE"
# read dataset
quantile_dataset_file = quantile_location / "preprocessing_dataset.yaml"
with open(quantile_dataset_file, "r", encoding="utf8") as f:
    quantile_dataset = yaml.load(f, yaml.UnsafeLoader)

# %%

results_all_list = []

for location in [
    "all",
    "Frankfurt",
    "Regensburg",
    "Mannheim",
    "Not-Frankfurt",
    "Not-Regensburg",
    "Not-Mannheim",
]:

    print(f"Starting with {location}.")

    experiment_group_name = f"Normalization_{location}"

    results_list = []
    dataset_file = (
        exp_group_base_dir
        / experiment_group_name
        / "data_preprocessed"
        / "GAN_DISCRIMINATORS"
        / "preprocessing_dataset.yaml"
    )
    # read dataset
    analysis_file = dataset_file.parent / "analysis.csv"
    if analysis_file.exists():
        results_location = pd.read_csv(analysis_file, sep=";", index_col=0)
    else:
        if not dataset_file.exists():
            print(f"{location} not yet finished.")
            continue
        with open(dataset_file, "r", encoding="utf8") as f:
            dataset = yaml.load(f, yaml.UnsafeLoader)

        for pat_name, data in tqdm(dataset.items()):
            image_gan = experiment_dir / data["image"]
            image_quant = experiment_dir / quantile_dataset[pat_name]["image"]

            for channel in range(3):
                result_metrics = evaluation.evaluate_autoencoder_prediction(
                    str(image_gan), str(image_quant), channel=channel
                )
                result_metrics["modality"] = channel
                result_metrics["location"] = location
                result_metrics["patient_id"] = pat_name

                results_list.append(result_metrics)

        results_location = pd.DataFrame(results_list)
        results_location.to_csv(analysis_file, sep=";")

    results_all_list.append(results_location)

results = pd.concat(results_all_list).reset_index().drop(columns="index")

# %%

sns.catplot(data=results, y="rmse", kind="box", row="location", col="modality")
plt.show()
plt.close()

sns.catplot(
    data=results,
    y="structured_similarity_index",
    kind="box",
    row="location",
    col="modality",
)
plt.show()
plt.close()

# %%

mean_results = results.groupby(["patient_id", "modality"]).median()

display(mean_results.sort_values("structured_similarity_index"))
