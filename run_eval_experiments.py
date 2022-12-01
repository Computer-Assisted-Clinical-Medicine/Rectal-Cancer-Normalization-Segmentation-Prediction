"""Run all experiments using the experiments file
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from run_single_experiment import eval_fold
from SegClassRegBasis.experiment import Experiment

experiment_dir = Path(os.environ["experiment_dir"])


def get_experiments():
    """Get all experiments and see which are completed"""
    experiments_file = experiment_dir / "Normalization_Experiment" / "experiments.json"
    exp_df = pd.read_json(experiments_file)
    exp_df["completed"] = False

    for exp_id, exp_row in exp_df.iterrows():
        exp_path = experiment_dir / exp_row.path
        if exp_path.exists():
            result_dirs_exist = []
            for task in exp_row.tasks.values():
                versions = exp_row.versions
                if task == "segmentation":
                    versions += [f"{ver}-postprocessed" for ver in versions]
                for ver in versions:
                    names = ["test"]
                    if exp_row.external:
                        names.append("external_testset")
                    for exp_name in names:
                        result_dirs_exist.append(
                            (exp_path / f"results_{exp_name}_{ver}_{task}").exists()
                        )
            if np.all(result_dirs_exist):
                print(f"{exp_row.path} already finished with training and evaluated.")
                exp_df.loc[exp_id, "completed"] = True
    return exp_df


experiments = get_experiments()
num_completed = experiments.completed.sum()
message = f"{num_completed} of {experiments.shape[0]} experiments already finished."
print(message)

# always load all experiments again, because sometimes experiments are prepared
# during the training
experiments = get_experiments()
num_completed = experiments.completed.sum()
for num, (_, exp_pd) in enumerate(experiments[~experiments.completed].iterrows()):
    print(f"Starting with {exp_pd.path}")
    # load experiment
    param_file = experiment_dir / exp_pd.path / "parameters.yaml"
    if not param_file.exists():
        print("Parameter file not found, experiment will be skipped")
        continue
    try:
        exp = Experiment.from_file(param_file)
    except FileNotFoundError:
        print("Files are missing for this experiment")
        continue
    for f in range(exp.folds):
        print(f"Starting with fold {f} ({f+1}/{exp.folds})")
        try:
            eval_fold(exp, f)
        except FileNotFoundError:
            pass
        except Exception as exc:  # pylint:disable=broad-except
            message = f"{exp_pd.path} failed with {exc}"
            print(message)
