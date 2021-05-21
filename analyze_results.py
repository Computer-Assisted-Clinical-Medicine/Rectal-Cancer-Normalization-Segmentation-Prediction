# %% [markdown]
'''
# Analyze the trained network
## Import and Definitions
'''

import os
from pathlib import Path
import sys

import matplotlib
import numpy as np
import seaborn as sns

from utils import gather_results

# if on cluster, use other backend
# pylint: disable=wrong-import-position, ungrouped-imports, wrong-import-order
if 'CLUSTER' in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# pylint: disable=pointless-string-statement

# %% [markdown]
'''
## Set Paths
'''

data_dir = Path(os.environ['data_dir'])
experiment_dir = Path(os.environ['experiment_dir'])
plot_dir = experiment_dir / 'plots'

if not plot_dir.exists():
    plot_dir.mkdir()

def save_and_show(name:str):
    """Save the plot and if running in interactive mode, show the plot.

    Parameters
    ----------
    name : str
        The name to use when saving
    """
    # plt.savefig(plot_dir / f'{name}.pdf', dpi=600, facecolor='w')
    plt.savefig(plot_dir / f'{name}.png', dpi=600, facecolor='w')
    if sys.flags.interactive:
        plt.show()
    plt.close()

def shorten_names(dataframe):
    # make nicer names
    for delete in ['DICE-Res-', 'nBN-DO-', '-100']:
        dataframe.name = dataframe.name.str.replace(delete,'')

# %% [markdown]
'''
## Do the analysis for the training set
'''

results = gather_results(experiment_dir, combined=True)
# 1035_1 is with fat supression
results = results.drop(results.index[results['File Number'] == '1035_1'])
# make nicer names

shorten_names(results)

sns.catplot(data=results, y='name', x='Dice', kind='box', aspect=2)
save_and_show('test_set_dice_models')

sns.catplot(data=results, y='name', x='Dice', hue='fold', kind='box', aspect=2)
save_and_show('test_set_dice_models_folds')

sns.catplot(data=results, y='fold', x='Dice', kind='box', aspect=2)
save_and_show('test_set_dice_folds')

sns.catplot(data=results, y='File Number', x='Dice', kind='box', aspect=.3, height=15)
save_and_show('test_set_dice_files')

results_mean = results.groupby('File Number').mean()
plt.scatter(x=results_mean['Volume (L)'],y=results_mean['Dice'])
plt.xlabel('GT Volume')
plt.ylabel('Dice')
save_and_show('test_set_volume_vs_dice')

plt.hist([results_mean['Volume (L)'].values, results_mean['Volume (P)'].values])
plt.xlabel('Volume')
plt.ylabel('Occurrence')
plt.legend(labels=['Ground Truth', 'Predicted'])
save_and_show('test_set_label_volume_hist')

plt.scatter(x=results_mean['Volume (L)'],y=results_mean['Hausdorff'])
plt.xlabel('GT Volume')
plt.ylabel('Hausdorff')
save_and_show('test_set_volume_vs_hausdorff')

plt.scatter(x=results_mean['Volume (L)'],y=results_mean['Volume (P)'])
max_l = results_mean['Volume (L)'].max()
max_p = results_mean['Volume (P)'].max()
plt.plot([0, max_l], [0, max_p], color='gray')
plt.xlabel('GT Volume')
plt.ylabel('Predicted Volume')
save_and_show('test_set_volume_vs_volume')

sns.pairplot(results[['name', 'Dice', 'Hausdorff', 'Volume (P)']], hue='name')
save_and_show('test_model_pairplot')

# %% [markdown]
'''
## Do the analysis for the test-set
'''

data_dir = Path(os.environ['data_dir'])
experiment_dir = Path(os.environ['experiment_dir'])

results_ex = gather_results(experiment_dir, combined=True, external=True)
results_ex = results_ex[np.logical_not(results_ex['File Number'].str.startswith('99'))]
shorten_names(results_ex)

sns.catplot(data=results_ex, y='name', x='Dice', kind='box', aspect=2)
save_and_show('external_test_set_dice_models')

sns.catplot(data=results_ex, y='name', x='Dice', hue='fold', kind='box', aspect=2)
save_and_show('external_test_set_dice_models_folds')

sns.catplot(data=results_ex, y='fold', x='Dice', kind='box', aspect=2)
save_and_show('external_test_set_dice_folds')

sns.catplot(data=results_ex, y='File Number', x='Dice', hue='name', kind='box', aspect=1.4, height=6)
save_and_show('external_test_set_dice_files')

results_ex_mean = results_ex.groupby('File Number').mean()
plt.scatter(x=results_ex_mean['Volume (L)'],y=results_ex_mean['Dice'])
plt.xlabel('GT Volume')
plt.ylabel('Dice')
save_and_show('external_test_set_volume_vs_dice')

plt.scatter(x=results_ex_mean['Volume (L)'],y=results_ex_mean['Volume (P)'])
max_l = results_ex_mean['Volume (L)'].max()
max_p = results_ex_mean['Volume (P)'].max()
plt.plot([0, max_l], [0, max_p], color='gray')
plt.xlabel('GT Volume')
plt.ylabel('Predicted Volume')
save_and_show('external_test_set_volume_vs_volume')
