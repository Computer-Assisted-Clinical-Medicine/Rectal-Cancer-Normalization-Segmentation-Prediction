"""Check the normalization to see if there are any large deviations"""
# %%

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml
from IPython.display import display
from tqdm.autonotebook import tqdm

from networks import auto_encoder
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
            print(f"\t{location} not yet finished.")
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

sns.catplot(
    data=results,
    y="norm_mutual_inf",
    kind="box",
    row="location",
    col="modality",
)
plt.show()
plt.close()

# %%

mean_results = results.groupby(["patient_id", "modality"]).median()

display(
    mean_results.sort_values("structured_similarity_index")[
        ["rmse", "norm_mutual_inf", "structured_similarity_index"]
    ]
)

display(
    results.sort_values("structured_similarity_index")[
        ["patient_id", "location", "rmse", "norm_mutual_inf", "structured_similarity_index"]
    ]
)

# %%
# plot the generator


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

    for channel in range(3):
        exp_dir = (
            exp_group_base_dir
            / experiment_group_name
            / "Train_Normalization_GAN"
            / f"Train_Normalization_GAN_{channel}"
        )

        model_file = exp_dir / "generator_with_shapes.png"
        if model_file.exists():
            continue

        with open(exp_dir / "parameters.yaml", "r", encoding="utf8") as f:
            norm_settings = yaml.load(f, yaml.UnsafeLoader)

        hp_train = norm_settings["hyper_parameters"]["train_parameters"]
        hp_net = norm_settings["hyper_parameters"]["network_parameters"]
        model = auto_encoder(
            inputs=tf.keras.Input(
                (hp_train["in_plane_dimension"], hp_train["in_plane_dimension"], 1),
                batch_size=hp_train["batch_size"],
            ),
            depth=hp_net["depth"],
            filter_base=hp_net["filter_base"],
            skip_edges=hp_net["skip_edges"],
            output_min=hp_net.get("output_min", None),
            output_max=hp_net.get("output_max", None),
            variational=hp_net.get("variational", False),
        )
        tf.keras.utils.plot_model(
            model,
            to_file=model_file,
            show_shapes=True,
        )
