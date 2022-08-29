"""
Combine the images from multiple models
"""
import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.autonotebook import tqdm

import SegClassRegBasis.config as cfg
from seg_data_loader import ApplyLoader
from SegClassRegBasis import evaluation
from SegClassRegBasis.experiment import Experiment
from SegClassRegBasis.postprocessing import keep_big_structures

WRITE_PROBABILITIES = False
OVERWRITE = False


def init_argparse():
    """
    initialize the parser
    """
    argpar = argparse.ArgumentParser(
        description="Combine the labels from different experiments."
    )
    argpar.add_argument(
        "-p",
        "--path",
        metavar="path",
        type=str,
        nargs="?",
        help="The directory of the experiment to combine.",
    )
    return argpar


def calculate_ensemble_weights(
    experiments: List[Experiment], metric="Dice", version="best"
) -> pd.DataFrame:
    """Calculate the weights each individual model and fold should have according
    to the metric (higher is assumed to be better)

    Parameters
    ----------
    experiments : List
        List of experiments
    metric : str, optional
        The metric to use, by default 'Dice'
    version : str, optional
        Which version of the model should be used, by default "best"

    Returns
    -------
    pd.DataFrame
        The resulting weights with columns for the total_weight, model_weight and
        fold_weight

    Raises
    ------
    FileNotFoundError
        [description]
    """
    models_list = []
    # collect all model results
    for exp in experiments:
        out_path = exp.output_path
        result_path = out_path / f"results_test_{version}-postprocessed"
        result_file = (
            result_path / f"evaluation-mean-results_test_{version}-postprocessed.csv"
        )
        if not result_file.exists():
            raise FileNotFoundError("Not all models are finished yet.")
        results: pd.DataFrame = pd.read_csv(result_file, sep=";")
        assert results.shape[0] > 0, f"The result file is empty ({result_file})"
        for number, row in results.iterrows():
            models_list.append(
                {
                    "model": exp.name,
                    "fold": exp.fold_dir_names[number],
                    "metric": row[metric],
                    "fold_dir": out_path / exp.fold_dir_names[number],
                }
            )
    models = pd.DataFrame(models_list)
    # determine fold weight
    fold_mean = models.groupby("fold")["metric"].mean()
    fold_weight = fold_mean / fold_mean.sum()
    # write to model
    for fold, weight in fold_weight.items():
        models.loc[models.fold == fold, "fold_weight"] = weight
    # determine model weight
    model_mean = models.groupby("model")["metric"].mean()
    model_weight = model_mean / model_mean.sum()
    # write to model
    for model, weight in model_weight.items():
        models.loc[models.model == model, "model_weight"] = weight
    # get the total weight
    models["total_weight"] = models.fold_weight * models.model_weight
    return models


def combine_models(
    experiments: List[Experiment],
    patients: Iterable,
    weights: pd.DataFrame,
    result_path: Path,
    name: str,
    overwrite: bool,
    version="best",
):
    """Combine the models using the provided weights

    Parameters
    ----------
    experiments : List
        List of experiments
    patients : Iterable
        The list of patients as list of paths
    weights : pd.DataFrame
        Dataframe with the weights, the total_weight column is used
    result_path : Path
        The path where the results should be saved
    name : str
        The name that should be used as model name
    overwrite : bool
        If existing files should be overwritten
    version : str, optional
        The version of the images to use, by default "best"
    """

    # remember the results
    results_list = []
    results_post_list = []
    p_id = None

    fold = result_path.parent.name
    eval_file_path = result_path.parent / f"evaluation-{fold}-{version}_{name}.csv"
    eval_file_path_post = (
        result_path.parent / f"evaluation-{fold}-{version}-postprocessed_{name}.csv"
    )

    if eval_file_path.exists() and eval_file_path_post.exists() and not overwrite:
        tqdm.write("Already finished")
        return

    # set preprocessing dir
    cfg.data_base_dir = experiments[0].experiment_dir

    for pat in tqdm(patients, unit="patient"):

        if not result_path.exists():
            result_path.mkdir(parents=True)

        # define paths
        p_id = Path(pat).name
        pred_path = result_path / f"prediction-{p_id}-{version}{cfg.file_suffix}"
        pred_path_post = (
            result_path / f"prediction-{p_id}-{version}-postprocessed{cfg.file_suffix}"
        )

        # load reference images
        testloader = ApplyLoader(
            name="test_loader",
            file_dict=experiments[0].data_set,
        )

        # see if it should be overwritten
        if not pred_path.exists() or overwrite:

            # find all predictions
            p_files = weights.fold_dir.apply(
                lambda x: x
                / result_path.name
                / f"prediction-{p_id}-{version}{cfg.file_suffix}"
            )
            found = p_files.apply(lambda x: x.exists())
            # skip files were nothing was found (they are probably in a different fold)
            if not np.any(found):
                continue
            p_weights = weights.total_weight[found]
            # norm them
            p_weights = p_weights / p_weights.sum()
            probability_files = p_files[found]

            # read and average the probabilities
            probability_avg = None
            first_image = None
            for prop_file, weight in zip(probability_files, p_weights):
                try:
                    image = sitk.ReadImage(str(prop_file))
                except ValueError as exc:
                    print(f"There was an error reading {prop_file}")
                    print(exc)
                if first_image is None:
                    first_image = image
                else:
                    # resample if there is some mismatch
                    if not np.all(
                        [
                            np.allclose(image.GetSize(), first_image.GetSize()),
                            np.allclose(
                                image.GetOrigin(), first_image.GetOrigin(), atol=0.01
                            ),
                            np.allclose(
                                image.GetDirection(), first_image.GetDirection(), atol=0.01
                            ),
                        ]
                    ):
                        print(f"{pat} was resample because of a size miss-match.")
                        image = sitk.Resample(image, referenceImage=first_image)
                labels = sitk.GetArrayFromImage(image)
                if probability_avg is None:
                    probability_avg = labels * weight
                else:
                    probability_avg += labels * weight
            assert probability_avg is not None, "No probabilities found"

            # write probabilities
            if WRITE_PROBABILITIES:
                with open(result_path / f"prediction-{p_id}-{version}.npz", "wb") as file:
                    np.savez_compressed(file, probability_avg)

            cfg.data_base_dir = experiments[0].experiment_dir
            ref_img_pre = testloader.get_processed_image(pat)
            original_image = testloader.get_original_image(pat)

            # generate labels
            predicted_labels = np.round(probability_avg).astype(int)
            # make it into an image
            predicted_label_img = sitk.GetImageFromArray(predicted_labels)
            predicted_label_img.CopyInformation(ref_img_pre)

            # resample to the original file
            predicted_label_orig = sitk.Resample(
                image1=predicted_label_img,
                referenceImage=original_image,
                interpolator=sitk.sitkNearestNeighbor,
                outputPixelType=sitk.sitkUInt8,
                useNearestNeighborExtrapolator=False,
            )

            sitk.WriteImage(predicted_label_orig, str(pred_path.resolve()))

            # postprocess
            keep_big_structures(pred_path, pred_path_post)

        # evaluate
        label_path = testloader.get_filenames(pat)[1]
        if label_path is not None:
            if not Path(label_path).exists():
                print(f"{label_path} not found")
                continue
            result_metrics = {"File Number": p_id}
            result_metrics.update(
                evaluation.evaluate_segmentation_prediction(str(pred_path), str(label_path))
            )
            results_list.append(result_metrics)

            result_metrics = {"File Number": p_id}
            result_metrics.update(
                evaluation.evaluate_segmentation_prediction(
                    str(pred_path_post), str(label_path)
                )
            )
            results_post_list.append(result_metrics)

    # write evaluation results
    if len(results_list) > 0:
        results = pd.DataFrame(results_list)
        results.set_index("File Number", inplace=True)
        results.to_csv(eval_file_path, sep=";")
        # also the postprocessed ones
        results_post = pd.DataFrame(results_post_list)
        results_post.set_index("File Number", inplace=True)
        results_post.to_csv(eval_file_path_post, sep=";")


def run_combine(experiments: List[Experiment], version="best"):
    """
    Run the combination including the analysis at the end.
    """
    # get the weights
    ensemble_weights = calculate_ensemble_weights(
        experiments, metric="Dice", version=version
    )

    work_dir = experiment_dir / "combined_models"
    if not work_dir.exists():
        work_dir.mkdir()

    # get the dataset
    data_set = experiments[0].data_set
    for exp in experiments:
        assert np.all(
            exp.data_set.keys() == data_set.keys()
        ), f"Not the same data set for {exp.name}"

    # see if there is an external test-set specified
    external_set_present = [
        getattr(e, "external_test_set") is not None for e in experiments
    ]
    if np.any(external_set_present):
        assert np.all(external_set_present), "external set was not used for all models"
        external_test_set = experiments[0].external_test_set
        for exp in experiments:
            assert np.all(
                exp.external_test_set == external_test_set
            ), f"Not the same data set for {exp.name}"
    else:
        external_test_set = np.array([])
    assert isinstance(external_test_set, np.ndarray)

    # do the predictions for the train set
    eval_files = []
    eval_files_post = []
    for f, weight in ensemble_weights.groupby("fold"):
        tqdm.write(f)
        combine_models(
            experiments,
            data_set,
            weight,
            work_dir / f / "apply",
            "test",
            overwrite=OVERWRITE,
            version=version,
        )
        eval_files.append(work_dir / f / f"evaluation-{f}-{version}_test.csv")
        eval_files_post.append(
            work_dir / f / f"evaluation-{f}-{version}-postprocessed_test.csv"
        )
    evaluation.combine_evaluation_results_from_folds(
        work_dir / f"results_test_{version}", eval_files
    )
    evaluation.combine_evaluation_results_from_folds(
        work_dir / f"results_test_{version}-postprocessed", eval_files_post
    )

    # and also for the external set
    if external_test_set.size > 0:
        name = "external_testset"
        tqdm.write(name)
        eval_files = []
        eval_files_post = []
        for f, weight in ensemble_weights.groupby("fold"):
            tqdm.write(f)
            applied = work_dir / f / "apply_external_testset"
            combine_models(
                experiments,
                external_test_set,
                weight,
                applied,
                name,
                overwrite=OVERWRITE,
                version=version,
            )
            eval_files.append(work_dir / f / f"evaluation-{f}-{version}_{name}.csv")
            eval_files_post.append(
                work_dir / f / f"evaluation-{f}-{version}-postprocessed_{name}.csv"
            )
        # combine all folds
        f = "all-folds"
        tqdm.write(f)
        weight = ensemble_weights
        applied = work_dir / f / "apply_external_testset"
        combine_models(
            experiments,
            external_test_set,
            weight,
            applied,
            name,
            overwrite=OVERWRITE,
            version=version,
        )
        eval_files.append(work_dir / f / f"evaluation-{f}-{version}_{name}.csv")
        eval_files_post.append(
            work_dir / f / f"evaluation-{f}-{version}-postprocessed_{name}.csv"
        )
        evaluation.combine_evaluation_results_from_folds(
            work_dir / f"results_{name}_{version}", eval_files
        )
        evaluation.combine_evaluation_results_from_folds(
            work_dir / f"results_{name}_{version}-postprocessed", eval_files_post
        )


if __name__ == "__main__":

    parser = init_argparse()
    args = parser.parse_args()
    experiment_dir = Path(args.path)

    hparam_file = experiment_dir / "hyperparameters.csv"
    all_experiments = []
    for e in pd.read_csv(hparam_file, sep=";")["path"]:
        param_file = experiment_dir.parent / Path(e) / "parameters.yaml"
        all_experiments.append(Experiment.from_file(param_file))

    # combine the models
    for vers in all_experiments[0].versions:
        run_combine(experiments=all_experiments, version=vers)
