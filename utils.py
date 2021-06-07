'''
Miscellaneous functions used mainly for plotting
'''
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from SegmentationNetworkBasis.architecture import DenseTiramisu, UNet

# if on cluster, use other backend
# pylint: disable=wrong-import-position, ungrouped-imports
if 'CLUSTER' in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt # pylint: disable=wrong-import-order


def plot_hparam_comparison(hparam_dir, metrics=None, external=False, postprocessed=False):
    '''
    Do separate plots for every changed hyperparameter.
    '''

    if metrics is None:
        metrics = ['Dice']

    hparam_file = hparam_dir / 'hyperparameters.csv'
    hparam_changed_file = hparam_dir / 'hyperparameters_changed.csv'

    if external:
        file_field = 'results_file_external_testset'
        result_name = 'hyperparameter_comparison_external_testset'
    else:
        file_field = 'results_file'
        result_name = 'hyperparameter_comparison'

    if postprocessed:
        file_field += '_postprocessed'
        result_name += '_postprocessed'

    # add pdf
    result_name += '.pdf'

    hparams = pd.read_csv(hparam_file, sep=';')
    hparams_changed = pd.read_csv(hparam_changed_file, sep=';')
    changed_params = hparams_changed.columns[1:]
    # collect all results
    results_means = []
    results_stds = []
    for results_file in hparams[file_field]:
        if Path(results_file).exists():
            results = pd.read_csv(results_file, sep=';')
            # save results
            results_means.append(results[metrics].mean())
            results_stds.append(results[metrics].std())
        else:
            name = Path(results_file).parent.parent.name
            print(f'Could not find the evaluation file for {name}'
                +' (probably not finished with training yet).')
            results_means.append(pd.Series({m : pd.NA for m in metrics}))
            results_stds.append(pd.Series({m : pd.NA for m in metrics}))

    if len(results_means) == 0:
        print('No files to evaluate')
        return

    # convert to dataframes
    results_means = pd.DataFrame(results_means)
    results_stds = pd.DataFrame(results_stds)

    # plot all metrics with all parameters
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(changed_params),
        sharey=True,
        figsize=(4*len(changed_params),6*len(metrics))
    )
    # fix the dimensions
    axes = np.array(axes).reshape((len(metrics), len(changed_params)))

    for met, ax_row in zip(metrics, axes):
        for col, ax in zip(changed_params, ax_row):
            # group by the other values
            unused_columns = [cn for cn in changed_params if col != cn]
            # if there are no unused columns, use the changed one
            if len(unused_columns) == 0:
                unused_columns = list(changed_params)
            for group, data in hparams_changed.groupby(unused_columns, dropna=False):
                # plot them with the same line
                # get the data
                m_data = results_means.loc[data.index,met]
                # sort by values
                m_data.sort_values()
                # only plot if not nan
                if not m_data.isna().all():
                    ax.plot(
                        data.loc[m_data.notna(), col], m_data[m_data.notna()],
                        marker='x',
                        label=str(group)
                    )
            # if the label is text, turn it
            if not pd.api.types.is_numeric_dtype(hparams_changed[col]):
                plt.setp(
                    ax.get_xticklabels(),
                    rotation=45,
                    ha='right',
                    rotation_mode='anchor'
                )
            # ylabel if it is the first image
            if col == changed_params[0]:
                ax.set_ylabel(met)
            # xlabel if it is the last row
            if met == metrics[-1]:
                ax.set_xlabel(col)
            # if the class is bool, replace the labels with the boolean values
            if isinstance(hparams_changed.iloc[0][col], np.bool_):
                ax.set_xticks([0,1])
                ax.set_xticklabels(['false', 'true'])

            # set the legend with title
            ax.legend(title = str(tuple(str(c)[:5] for c in unused_columns)))

    fig.suptitle('Hypereparameter Comparison')
    plt.tight_layout()
    plt.savefig(hparam_dir / result_name)
    plt.close()


def compare_hyperparameters(experiments, experiment_dir, version='best'):
    '''
    Compare the hyperparameters of all experiments and collect the ones that
    were changed.
    '''
    # export the hyperparameters
    hyperparameter_file = experiment_dir / 'hyperparameters.csv'
    hyperparameter_changed_file = experiment_dir / 'hyperparameters_changed.csv'
    # collect all results
    hparams = []
    for exp in experiments:
        res_name = 'evaluation-all-files.csv'
        res = exp.output_path / f'results_test_{version}' / res_name
        res_post = exp.output_path / f'results_test_{version}-postprocessed' / res_name
        res_ext = exp.output_path / f'results_external_testset_{version}' / res_name
        res_ext_post = exp.output_path / f'results_external_testset_{version}-postprocessed' / res_name
        # and parameters
        hparams.append({
            **exp.hyper_parameters['network_parameters'],
            **exp.hyper_parameters['train_parameters'],
            **exp.hyper_parameters['data_loader_parameters'],
            'loss' : exp.hyper_parameters['loss'],
            'architecture' : exp.hyper_parameters['architecture'].__name__,
            'dimensions' : exp.hyper_parameters['dimensions'],
            'path' : exp.output_path,
            'results_file' : res,
            'results_file_postprocessed' : res_post,
            'results_file_external_testset' : res_ext,
            'results_file_external_testset_postprocessed' : res_ext_post
        })

    # convert to dataframes
    hparams = pd.DataFrame(hparams)
    # find changed parameters
    changed_params = []
    # drop the results file when analyzing the changed hyperparameters
    for col in hparams.drop(columns='results_file'):
        if hparams[col].astype(str).unique().size > 1:
            changed_params.append(col)
    hparams_changed = hparams[changed_params].copy()
    # if n_filters, use the first
    if 'n_filters' in hparams_changed:
        hparams_changed.loc[:,'n_filters'] = hparams_changed['n_filters'].dropna().apply(lambda x: x[0])
    if 'normalizing_method' in hparams_changed:
        n_name = hparams_changed['normalizing_method'].apply(lambda x: x.name)
        hparams_changed.loc[:,'normalizing_method'] = n_name
    # ignore the batch size (it correlates with the dimension)
    if 'batch_size' in hparams_changed:
        hparams_changed.drop(columns='batch_size', inplace=True)
    # ignore do_bias (it is set the opposite to batch_norm)
    if 'do_bias' in hparams_changed and 'do_batch_normalization' in hparams_changed:
        hparams_changed.drop(columns='do_bias', inplace=True)
    # drop column specifying the files
    if 'path' in hparams_changed:
        hparams_changed.drop(columns='path', inplace=True)
    # drop column specifying the files
    if 'results_file_postprocessed' in hparams_changed:
        hparams_changed.drop(columns='results_file_postprocessed', inplace=True)
    # drop column specifying the files
    if 'results_file_external_testset' in hparams_changed:
        hparams_changed.drop(columns='results_file_external_testset', inplace=True)
    # drop column specifying the files
    if 'results_file_external_testset_postprocessed' in hparams_changed:
        hparams_changed.drop(columns='results_file_external_testset_postprocessed', inplace=True)
    # drop columns only related to architecture
    arch_params = hparams_changed.groupby('architecture').nunique(dropna=False)
    for col in arch_params:
        if np.all(arch_params[col] == 1):
            hparams_changed.drop(columns=col, inplace=True)


    hparams.to_csv(hyperparameter_file, sep=';')
    hparams_changed.to_csv(hyperparameter_changed_file, sep=';')


def generate_folder_name(parameters):
    '''
    Make a name summarizing the hyperparameters.
    '''
    epochs = parameters['train_parameters']['epochs']

    params = [
        parameters['architecture'].get_name() + str(parameters['dimensions']) + 'D',
        parameters['loss']
    ]

    # TODO: move this logic into the network
    if parameters['architecture'] is  UNet:
        # residual connections if it is an attribute
        if 'res_connect' in parameters['network_parameters']:
            if parameters['network_parameters']['res_connect']:
                params.append('Res')
            else:
                params.append('nRes')

        # filter multiplier
        params.append('f_'+str(parameters['network_parameters']['n_filters'][0]//8))

        # batch norm
        if parameters['network_parameters']['do_batch_normalization']:
            params.append('BN')
        else:
            params.append('nBN')

        # dropout
        if parameters['network_parameters']['drop_out'][0]:
            params.append('DO')
        else:
            params.append('nDO')
    elif parameters["architecture"] is DenseTiramisu:
        params.append('gr_'+str(parameters['network_parameters']['growth_rate']))

        params.append('nl_'+str(len(parameters['network_parameters']['layers_per_block'])))
    else:
        raise NotImplementedError(f'{parameters["architecture"]} not implemented')

    # normalization
    params.append(str(parameters['data_loader_parameters']['normalizing_method'].name))

    # object fraction
    params.append(f'obj_{int(parameters["train_parameters"]["percent_of_object_samples"]*100):03d}%')

    # add epoch number
    params.append(str(epochs))

    folder_name = "-".join(params)

    return folder_name


def gather_results(experiment_dir, external=False, postprocessed=False, combined=True, version='best')->pd.DataFrame:
    """Collect all result files from all experiments. Only experiments that are
    already finished will be included in the analysis.

    Parameters
    ----------
    experiment_dir : Pathlike
        The path where the experiments are located
    external : bool, optional
        If the external testset should be evaluated, by default False
    postprocessed : bool, optional
        If the data from the posprocessed should be evaluated, by default False
    combined : bool, optional
        If there is a combined model, which should be analyzed, by default True
    version : str, optional
        Which version of the model should be used, by default best

    Returns
    -------
    pd.DataFrame
        The results with all metrics for all files
    """
    hparam_file = experiment_dir / 'hyperparameters.csv'

    if external:
        file_field = 'results_file_external_testset'
    else:
        file_field = 'results_file'

    if postprocessed:
        file_field += '_postprocessed'

    hparams = pd.read_csv(hparam_file, sep=';')

    # add combined model if present
    if combined:
        c_path = experiment_dir / 'combined_models'
        r_file = 'evaluation-all-files.csv'
        loc = hparams.shape[0]
        hparams.loc[loc] = 'Combined'
        hparams.loc[loc, 'results_file'] = c_path
        hparams.loc[loc, 'results_file'] = c_path / f'results_{version}_test' / r_file
        hparams.loc[loc, 'results_file_postprocessed'] = c_path / f'results_{version}-postprocessed_test' / r_file
        hparams.loc[loc, 'results_file_external_testset'] = c_path / f'results_{version}_external_testset' / r_file
        r_path = c_path / f'results_{version}-postprocessed_external_testset' / r_file
        hparams.loc[loc, 'results_file_external_testset_postprocessed'] = r_path

    results_all_list = []
    for _, row in hparams.iterrows():
        results_file = row[file_field]
        if Path(results_file).exists():
            results = pd.read_csv(results_file, sep=';')
            # set the model
            results['name'] = Path(row['path']).name
            # save results
            results_all_list.append(results)
        else:
            name = Path(results_file).parent.parent.name
            print(f'Could not find the evaluation file for {name}'
                +' (probably not finished with training yet).')

    results_all = pd.concat(results_all_list)
    # drop first column (which is just the old index)
    results_all.drop(results_all.columns[0], axis='columns', inplace=True)
    results_all['fold'] = pd.Categorical(results_all['fold'])
    results_all['name'] = pd.Categorical(results_all['name'])
    results_all.index = pd.RangeIndex(results_all.shape[0])
    results_all.sort_values('File Number', inplace=True)
    return results_all
