# %% [markdown]
'''
# Analyze the trained network
## Import and Definitions
'''

import os
from pathlib import Path

import matplotlib

# if on cluster, use other backend
if 'CLUSTER' in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def gather_results(experiment_dir, external=False, postprocessed=False, combined=False):
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
        l = hparams.shape[0]
        hparams.loc[l] = 'Combined'
        hparams.loc[l, 'results_file'] = c_path
        hparams.loc[l, 'results_file'] = c_path / 'results_final_test' / r_file
        hparams.loc[l, 'results_file_postprocessed'] = c_path / 'results_final-postprocessed_test' / r_file
        hparams.loc[l, 'results_file_external_testset'] = c_path / 'results_final_external_testset' / r_file
        hparams.loc[l, 'results_file_external_testset_postprocessed'] = c_path / 'results_final-postprocessed_external_testset' / r_file

    results_all = []
    for _, row in hparams.iterrows():
        results_file = row[file_field]
        if Path(results_file).exists():
            results = pd.read_csv(results_file, sep=';')
            # set the model
            results['name'] = Path(row['path']).name
            # save results
            results_all.append(results)
        else:
            name = Path(results_file).parent.parent.name
            print(f'Could not find the evaluation file for {name}'
                +' (probably not finished with training yet).')

    # convert to dataframes
    results_all = pd.concat(results_all)
    # drop first column (which is just the old index)
    results_all.drop(results_all.columns[0], axis='columns', inplace=True)
    # make categoricals
    results_all['fold'] = pd.Categorical(results_all['fold'])
    results_all['name'] = pd.Categorical(results_all['name'])
    results_all.index = pd.RangeIndex(results_all.shape[0])
    return results_all


# %% [markdown]
'''
## Do the analysis for the training set
'''

data_dir = Path(os.environ['data_dir'])
experiment_dir = Path(os.environ['experiment_dir'])

results = gather_results(experiment_dir, combined=True)
# 1035_1 is with fat supression
results = results.drop(results.index[results['File Number'] == '1035_1'])

sns.catplot(data=results, y='name', x='Dice', kind='box', aspect=2)
plt.show()
plt.close()

sns.catplot(data=results, y='name', x='Dice', hue='fold', kind='box', aspect=2)
plt.show()
plt.close()

sns.catplot(data=results, y='fold', x='Dice', kind='box', aspect=2)
plt.show()
plt.close()

sns.catplot(data=results, y='File Number', x='Dice', kind='box', aspect=.3, height=15)
plt.show()
plt.close()

# %% [markdown]
'''
## Do the analysis for the test-set
'''

data_dir = Path(os.environ['data_dir'])
experiment_dir = Path(os.environ['experiment_dir'])

results_ex = gather_results(experiment_dir, combined=True, external=True)

sns.catplot(data=results_ex, y='name', x='Dice', kind='box', aspect=2)
plt.show()
plt.close()

sns.catplot(data=results_ex, y='name', x='Dice', hue='fold', kind='box', aspect=2)
plt.show()
plt.close()

sns.catplot(data=results_ex, y='fold', x='Dice', kind='box', aspect=2)
plt.show()
plt.close()

sns.catplot(data=results_ex, y='File Number', x='Dice', kind='box', aspect=1.4, height=4)
plt.show()
plt.close()
# %%
