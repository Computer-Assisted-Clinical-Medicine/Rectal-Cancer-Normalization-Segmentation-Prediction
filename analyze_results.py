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
    results_all.sort_values('File Number', inplace=True)
    return results_all

# %% [markdown]
'''
## Set Paths
'''

data_dir = Path(os.environ['data_dir'])
experiment_dir = Path(os.environ['experiment_dir'])
plot_dir = experiment_dir / 'plots'

if not plot_dir.exists():
    plot_dir.mkdir()

def save(name):
    # plt.savefig(plot_dir / f'{name}.pdf', dpi=600, facecolor='w')
    plt.savefig(plot_dir / f'{name}.png', dpi=600, facecolor='w')


# %% [markdown]
'''
## Do the analysis for the training set
'''

results = gather_results(experiment_dir, combined=True)
# 1035_1 is with fat supression
results = results.drop(results.index[results['File Number'] == '1035_1'])

sns.catplot(data=results, y='name', x='Dice', kind='box', aspect=2)
save('results_dice_models')
plt.show()
plt.close()

sns.catplot(data=results, y='name', x='Dice', hue='fold', kind='box', aspect=2)
save('results_dice_models_folds')
plt.show()
plt.close()

sns.catplot(data=results, y='fold', x='Dice', kind='box', aspect=2)
save('results_dice_folds')
plt.show()
plt.close()

sns.catplot(data=results, y='File Number', x='Dice', kind='box', aspect=.3, height=15)
save('results_dice_files')
plt.show()
plt.close()

results_mean = results.groupby('File Number').mean()
plt.scatter(x=results_mean['Volume (L)'],y=results_mean['Dice'])
plt.xlabel('GT Volume')
plt.ylabel('Dice')
save('results_volume_vs_dice')
plt.show()
plt.close()

results_mean = results.groupby('File Number').mean()
plt.scatter(x=results_mean['Volume (L)'],y=results_mean['Hausdorff'])
plt.xlabel('GT Volume')
plt.ylabel('Hausdorff')
save('results_volume_vs_hausdorff')
plt.show()
plt.close()

plt.scatter(x=results_mean['Volume (L)'],y=results_mean['Volume (P)'])
max_l = results_mean['Volume (L)'].max()
max_p = results_mean['Volume (P)'].max()
plt.plot([0, max_l], [0, max_p], color='gray')
plt.xlabel('GT Volume')
plt.ylabel('Predicted Volume')
save('results_volume_vs_volume')
plt.show()
plt.close()

# %% [markdown]
'''
## Do the analysis for the test-set
'''

data_dir = Path(os.environ['data_dir'])
experiment_dir = Path(os.environ['experiment_dir'])

results_ex = gather_results(experiment_dir, combined=True, external=True)
results_ex = results_ex[np.logical_not(results_ex['File Number'].str.startswith('99'))]

sns.catplot(data=results_ex, y='name', x='Dice', kind='box', aspect=2)
save('external_results_dice_models')
plt.show()
plt.close()

sns.catplot(data=results_ex, y='name', x='Dice', hue='fold', kind='box', aspect=2)
save('external_results_dice_models_folds')
plt.show()
plt.close()

sns.catplot(data=results_ex, y='fold', x='Dice', kind='box', aspect=2)
save('external_results_dice_folds')
plt.show()
plt.close()

sns.catplot(data=results_ex, y='File Number', x='Dice', hue='name', kind='box', aspect=1.4, height=6)
save('external_results_dice_files')
plt.show()
plt.close()

results_ex_mean = results_ex.groupby('File Number').mean()
plt.scatter(x=results_ex_mean['Volume (L)'],y=results_ex_mean['Dice'])
plt.xlabel('GT Volume')
plt.ylabel('Dice')
save('external_results_volume_vs_dice')
plt.show()
plt.close()

plt.scatter(x=results_ex_mean['Volume (L)'],y=results_ex_mean['Volume (P)'])
max_l = results_ex_mean['Volume (L)'].max()
max_p = results_ex_mean['Volume (P)'].max()
plt.plot([0, max_l], [0, max_p], color='gray')
plt.xlabel('GT Volume')
plt.ylabel('Predicted Volume')
save('external_results_volume_vs_volume')
plt.show()
plt.close()

# %% [markdown]
'''
## Do the analysis for the data from Barbara
'''


data_dir = Path(os.environ['data_dir'])
experiment_dir = Path(os.environ['experiment_dir'])

results_ex_b = gather_results(experiment_dir, combined=True, external=True)
results_ex_b = results_ex_b[results_ex_b['File Number'].str.startswith('99')]

sns.catplot(data=results_ex_b, y='name', x='Dice', kind='box', aspect=2)
save('results_barbara_dice_models')
plt.show()
plt.close()

sns.catplot(data=results_ex_b, y='name', x='Dice', hue='fold', kind='box', aspect=2)
save('results_barbara_dice_models_folds')
plt.show()
plt.close()

sns.catplot(data=results_ex_b, y='fold', x='Dice', kind='box', aspect=2)
save('results_barbara_dice_folds')
plt.show()
plt.close()

sns.catplot(data=results_ex_b, y='File Number', x='Dice', hue='name', kind='box', aspect=1.4, height=6)
save('results_barbara_dice_files')
plt.show()
plt.close()

results_ex_b_mean = results_ex_b.groupby('File Number').mean()
plt.scatter(x=results_ex_b_mean['Volume (L)'],y=results_ex_b_mean['Dice'])
plt.xlabel('GT Volume')
plt.ylabel('Dice')
save('results_barbara_volume_vs_dice')
plt.show()
plt.close()

plt.scatter(x=results_ex_b_mean['Volume (L)'],y=results_ex_b_mean['Volume (P)'])
max_l = results_ex_b_mean['Volume (L)'].max()
max_p = results_ex_b_mean['Volume (P)'].max()
plt.plot([0, max_l], [0, max_p], color='gray')
plt.xlabel('GT Volume')
plt.ylabel('Predicted Volume')
save('results_barbara_volume_vs_volume')
plt.show()
plt.close()