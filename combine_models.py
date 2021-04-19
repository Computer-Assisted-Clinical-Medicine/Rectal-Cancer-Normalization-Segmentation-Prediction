from experiment import Experiment
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.autonotebook import tqdm

import evaluation
import SegmentationNetworkBasis.config as cfg
from seg_data_loader import ApplyLoader
from SegmentationNetworkBasis.postprocessing import keep_big_structures

write_probabilities = False

def calculate_ensemble_weights(experiments, metric='Dice'):
    models = []
    # collect all model results
    for e in experiments:
        out_path = e.output_path
        result_path = out_path / 'results_test_final-postprocessed'
        result_file = result_path / 'evaluation-mean-results_test_final-postprocessed.csv'
        results = pd.read_csv(result_file)
        for number, r in results.iterrows():
            models.append({
                'model' : e.name,
                'fold' : e.fold_dir_names[number],
                'metric' : r[metric],
                'fold_dir' : out_path / e.fold_dir_names[number]
            })
    models = pd.DataFrame(models)
    # determine fold weight
    fold_mean = models.groupby('fold')['metric'].mean()
    fold_weight = fold_mean / fold_mean.sum()
    # write to model
    for f, w in fold_weight.items():
        models.loc[models.fold==f, 'fold_weight'] = w
    # determine model weight
    model_mean = models.groupby('model')['metric'].mean()
    model_weight = model_mean / model_mean.sum()
    # write to model
    for m, w in model_weight.items():
        models.loc[models.model==m, 'model_weight'] = w
    # get the total weight
    models['total_weight'] = models.fold_weight * models.model_weight
    return models

if __name__ == '__main__':

    data_dir = Path(os.environ['data_dir'])
    experiment_dir = Path(os.environ['experiment_dir'])

    hparam_file = experiment_dir / 'hyperparameters.csv'
    experiments = []
    for e in pd.read_csv(hparam_file)['path']:
        param_file = Path(e) / 'parameters.yaml'
        experiments.append(Experiment.from_file(param_file))

    # get the weights
    weights = calculate_ensemble_weights(experiments, metric='Dice')
    version = 'final'

    work_dir = experiment_dir / 'combined_models' / 'apply'
    if not work_dir.exists():
        work_dir.mkdir()

    # get the dataset
    data_set = experiments[0].data_set
    for e in experiments:
        assert np.all(e.data_set==data_set), f'Not the same data set for {e.name}'

    # see if there is an external test-set specified
    external_set_present = [hasattr(e, 'external_test_set') for e in experiments]
    if np.any(external_set_present):
        assert np.all(external_set_present), 'external set was not used for all models'
        external_test_set = experiments[0].external_test_set
        for e in experiments:
            assert np.all(e.external_test_set==external_test_set), f'Not the same data set for {e.name}'
    else:
        external_test_set = []

    # remember the results
    results = []
    results_post = []
    # apply it for all patients
    for p in tqdm(np.concatenate([data_set, external_test_set]), unit='image'):

        if p in data_set:
            apply_name = 'apply'
        else:
            apply_name = 'apply_external_testset'
        result_path = work_dir / apply_name
        if not result_path.exists():
            result_path.mkdir()

        # define paths
        p_id = Path(p).name
        pred_path = result_path / f'prediction-{p_id}-{version}.nrrd'
        pred_path_post = result_path / f'prediction-{p_id}-{version}-postprocessed.nrrd'

        # find all predictions
        func = lambda x: (x / apply_name / f'prediction-{p_id}-{version}.npy')
        p_files = weights.fold_dir.apply(func)
        found = p_files.apply(lambda x: x.exists())
        p_weights = weights.total_weight[found]
        probability_files = p_files[found]
        probabilities = np.array([np.load(f) for f in probability_files])
        probability_avg = np.average(probabilities, axis=0, weights=p_weights)
        # write probabilities
        if write_probabilities:
            with open(result_path / f'prediction-{p_id}-{version}.npy', 'wb') as f:
                np.save(f, probability_avg)

        cfg.preprocessed_dir = experiments[0].preprocessed_dir
        # load reference images
        testloader = ApplyLoader(
            name='test_loader',
            **experiments[0].hyper_parameters['data_loader_parameters']
        )
        ref_img_pre = testloader.get_processed_image(experiments[0].data_dir / p)
        original_image = testloader.get_original_image(experiments[0].data_dir / p)

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

        pred_path = result_path / f'prediction-{p_id}-{version}.nrrd'
        sitk.WriteImage(predicted_label_orig, str(pred_path.resolve()))

        # postprocess
        keep_big_structures(pred_path, pred_path_post)

        # evaluate
        label_path = testloader._get_filenames(experiments[0].data_dir / p)[1]
        if Path(label_path).exists():
            result_metrics = {'File Number' : p_id}
            evaluation.evaluate_segmentation_prediction(
                result_metrics,
                str(pred_path),
                str(label_path)
            )
            results.append(result_metrics)

            result_metrics = {'File Number' : p_id}
            evaluation.evaluate_segmentation_prediction(
                result_metrics,
                str(pred_path_post),
                str(label_path)
            )
            results_post.append(result_metrics)

    # write evaluation results
    results = pd.DataFrame(results)
    results.set_index('File Number', inplace=True)
    eval_file_path = work_dir.parent / f'evaluation-all_models-{version}_test.csv'
    results.to_csv(eval_file_path, sep=';')
    # also the postprocessed ones
    results_post = pd.DataFrame(results_post)
    results_post.set_index('File Number', inplace=True)
    eval_file_path_post = work_dir.parent / f'evaluation-all_models-{version}-postprocessed_test.csv'
    results_post.to_csv(eval_file_path_post, sep=';')
