'''
Combine the images from multiple models
'''
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.autonotebook import tqdm

import evaluation
import SegmentationNetworkBasis.config as cfg
from experiment import Experiment
from seg_data_loader import ApplyLoader
from SegmentationNetworkBasis.postprocessing import keep_big_structures

WRITE_PROBABILITIES = False
OVERWRITE = False

def calculate_ensemble_weights(experiments:List, metric='Dice', version='best')->pd.DataFrame:
    """Calculate the weights each individual model and fold should have according
    to the metric (higher is assumed to be better)

    Parameters
    ----------
    experiments : List
        List of experiments
    metric : str, optional
        The metric to use, by default 'Dice'
    version : str, optional
        Which version of the model should be used, by default 'best'

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
        result_path = out_path / f'results_test_{version}-postprocessed'
        result_file = result_path / f'evaluation-mean-results_test_{version}-postprocessed.csv'
        if not result_file.exists():
            raise FileNotFoundError('Not all models are finished yet.')
        results = pd.read_csv(result_file, sep=';')
        assert results.shape[0] > 0, f'The result file is empty ({result_file})'
        for number, row in results.iterrows():
            models_list.append({
                'model' : exp.name,
                'fold' : exp.fold_dir_names[number],
                'metric' : row[metric],
                'fold_dir' : out_path / exp.fold_dir_names[number]
            })
    models = pd.DataFrame(models_list)
    # determine fold weight
    fold_mean = models.groupby('fold')['metric'].mean()
    fold_weight = fold_mean / fold_mean.sum()
    # write to model
    for fold, weight in fold_weight.items():
        models.loc[models.fold==fold, 'fold_weight'] = weight
    # determine model weight
    model_mean = models.groupby('model')['metric'].mean()
    model_weight = model_mean / model_mean.sum()
    # write to model
    for model, weight in model_weight.items():
        models.loc[models.model==model, 'model_weight'] = weight
    # get the total weight
    models['total_weight'] = models.fold_weight * models.model_weight
    return models

def combine_models(patients:Iterable, weights:pd.DataFrame, result_path:Path,
        name:str, overwrite:bool, version='best'):
    """Combine the models using the provided weights

    Parameters
    ----------
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
        The version of the images to use, by default 'best'
    """

    # remember the results
    results_list = []
    results_post_list = []

    for pat in tqdm(patients, unit='image'):

        if not result_path.exists():
            result_path.mkdir(parents=True)

        # define paths
        p_id = Path(pat).name
        pred_path = result_path / f'prediction-{p_id}-{version}.nrrd'
        pred_path_post = result_path / f'prediction-{p_id}-{version}-postprocessed.nrrd'

        # load reference images
        testloader = ApplyLoader(
            name='test_loader',
            **all_experiments[0].hyper_parameters['data_loader_parameters']
        )

        # see if it should be overwritten
        if not pred_path.exists() or overwrite:

            # find all predictions
            func = lambda x: (x / result_path.name / f'prediction-{p_id}-{version}.npy') # pylint: disable=cell-var-from-loop
            p_files = weights.fold_dir.apply(func)
            found = p_files.apply(lambda x: x.exists())
            # skip files were nothing was found (they are probably in a different fold)
            if not np.any(found):
                continue
            p_weights = weights.total_weight[found]
            probability_files = p_files[found]
            probabilities = np.array([np.load(f) for f in probability_files])
            probability_avg = np.average(probabilities, axis=0, weights=p_weights)
            # write probabilities
            if WRITE_PROBABILITIES:
                with open(result_path / f'prediction-{p_id}-{version}.npy', 'wb') as file:
                    np.save(file, probability_avg)

            cfg.preprocessed_dir = all_experiments[0].preprocessed_dir
            ref_img_pre = testloader.get_processed_image(all_experiments[0].data_dir / pat)
            original_image = testloader.get_original_image(all_experiments[0].data_dir / pat)

            # generate labels
            predicted_labels = np.argmax(probability_avg, -1)
            # make it into an image
            predicted_label_img = sitk.GetImageFromArray(predicted_labels)
            predicted_label_img.CopyInformation(ref_img_pre)

            # resample to the original file
            predicted_label_orig = sitk.Resample(
                image1=predicted_label_img,
                referenceImage=original_image,
                interpolator=sitk.sitkNearestNeighbor,
                outputPixelType=sitk.sitkUInt8,
                useNearestNeighborExtrapolator=True
            )

            sitk.WriteImage(predicted_label_orig, str(pred_path.resolve()))

            # postprocess
            keep_big_structures(pred_path, pred_path_post)

        # evaluate
        label_path = testloader._get_filenames(all_experiments[0].data_dir / pat)[1] # pylint: disable=protected-access
        if Path(label_path).exists():
            result_metrics = {'File Number' : p_id}
            evaluation.evaluate_segmentation_prediction(
                result_metrics,
                str(pred_path),
                str(label_path)
            )
            results_list.append(result_metrics)

            result_metrics = {'File Number' : p_id}
            evaluation.evaluate_segmentation_prediction(
                result_metrics,
                str(pred_path_post),
                str(label_path)
            )
            results_post_list.append(result_metrics)

    # write evaluation results
    fold = result_path.parent.name
    results = pd.DataFrame(results_list)
    results.set_index('File Number', inplace=True)
    eval_file_path = result_path.parent / f'evaluation-{fold}-{version}_{name}.csv'
    results.to_csv(eval_file_path, sep=';')
    # also the postprocessed ones
    results_post = pd.DataFrame(results_post_list)
    results_post.set_index('File Number', inplace=True)
    eval_file_path_post = result_path.parent / f'evaluation-{fold}-{version}-postprocessed_{name}.csv'
    results_post.to_csv(eval_file_path_post, sep=';')

if __name__ == '__main__':

    data_dir = Path(os.environ['data_dir'])
    experiment_dir = Path(os.environ['experiment_dir'])

    hparam_file = experiment_dir / 'hyperparameters.csv'
    all_experiments = []
    for e in pd.read_csv(hparam_file, sep=';')['path']:
        param_file = Path(e) / 'parameters.yaml'
        all_experiments.append(Experiment.from_file(param_file))

    # get the weights
    ensemble_weights = calculate_ensemble_weights(all_experiments, metric='Dice')
    VERSION = 'best'

    work_dir = experiment_dir / 'combined_models'
    if not work_dir.exists():
        work_dir.mkdir()

    # get the dataset
    data_set = all_experiments[0].data_set
    for e in all_experiments:
        assert np.all(e.data_set==data_set), f'Not the same data set for {e.name}'

    # see if there is an external test-set specified
    external_set_present = [hasattr(e, 'external_test_set') for e in all_experiments]
    if np.any(external_set_present):
        assert np.all(external_set_present), 'external set was not used for all models'
        external_test_set = all_experiments[0].external_test_set
        for e in all_experiments:
            assert np.all(e.external_test_set==external_test_set), f'Not the same data set for {e.name}'
    else:
        external_test_set = np.array([])

    # do the predictions for the train set
    eval_files = []
    eval_files_post = []
    for f, w in ensemble_weights.groupby('fold'):
        combine_models(data_set, w, work_dir / f / 'apply', 'test', overwrite=OVERWRITE)
        eval_files.append(work_dir / f / f'evaluation-{f}-{VERSION}_test.csv')
        eval_files_post.append(work_dir / f / f'evaluation-{f}-{VERSION}-postprocessed_test.csv')
    evaluation.combine_evaluation_results_from_folds(work_dir/f'results_{VERSION}_test', eval_files)
    evaluation.combine_evaluation_results_from_folds(work_dir/f'results_{VERSION}-postprocessed_test', eval_files_post)

    # and also for the external set
    if external_test_set.size > 0:
        NAME = 'external_testset'
        eval_files = []
        eval_files_post = []
        for f, w in ensemble_weights.groupby('fold'):
            applied = work_dir / f / 'apply_external_testset'
            combine_models(external_test_set, w, applied, NAME, overwrite=OVERWRITE)
            eval_files.append(work_dir / f / f'evaluation-{f}-{VERSION}_{NAME}.csv')
            eval_files_post.append(work_dir / f / f'evaluation-{f}-{VERSION}-postprocessed_{NAME}.csv')
        # combine all folds
        f = 'all-folds'
        w = ensemble_weights
        applied = work_dir / f / 'apply_external_testset'
        combine_models(external_test_set, w, applied, NAME, overwrite=OVERWRITE)
        eval_files.append(work_dir / f / f'evaluation-{f}-{VERSION}_{NAME}.csv')
        eval_files_post.append(work_dir / f / f'evaluation-{f}-{VERSION}-postprocessed_{NAME}.csv')
        evaluation.combine_evaluation_results_from_folds(
            work_dir/f'results_{VERSION}_{NAME}', eval_files
        )
        evaluation.combine_evaluation_results_from_folds(
            work_dir/f'results_{VERSION}-postprocessed_{NAME}', eval_files_post
        )
