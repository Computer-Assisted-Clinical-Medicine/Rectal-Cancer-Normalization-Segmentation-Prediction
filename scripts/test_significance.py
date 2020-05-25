from scipy import stats
import pandas as pd
import os
import numpy as np

folder_path = "R:\\vessel_results"

# u2_file = "evaluation-UNet2D-CEL+DICE-DO-nBN.csv"
# u3_file = "evaluation-UNet3D-CEL+DICE-DO-nBN.csv"
# v2_file = "evaluation-VNet2D-CEL+DICE-DO-nBN.csv"
# v3_file = "evaluation-VNet3D-CEL+DICE-DO-nBN.csv"
# d2_file = "evaluation-DVN2D-CEL+DICE-DO-nBN.csv"
# d3_file = "evaluation-DVN3D-CEL+DICE-DO-nBN.csv"
#
# files = [u2_file, u3_file, v2_file, v3_file, d2_file, d3_file]
# names = ["UNet2D", "UNet3D", "VNet2D", "VNet3D", "DVNet2D", "DVNet3D"]

# en1_file = "evaluation-Equal-CEL+DICE-DO-nBN.csv"
# en2_file = "evaluation-032D-CEL+DICE-DO-nBN.csv"
# en3_file = "evaluation-042D-CEL+DICE-DO-nBN.csv"
# en4_file = "evaluation-03UNet-CEL+DICE-DO-nBN.csv"
# en5_file = "evaluation-04UNet-CEL+DICE-DO-nBN.csv"
# en6_file = "evaluation-UNets-CEL+DICE-DO-nBN.csv"
# en7_file = "evaluation-VNets-CEL+DICE-DO-nBN.csv"
#
# files = [en1_file, en2_file, en3_file, en4_file, en5_file, en6_file, en7_file]
# names = ["Ensemble 1", "Ensemble 2", "Ensemble 3", "Ensemble 4", "Ensemble 5", "Ensemble 6", "Ensemble 7"]


u2_file = "evaluation-UNet2DBTCV-CEL+DICE-DO-nBN.csv"
u3_file = "evaluation-UNet3DBTCV-CEL+DICE-DO-nBN.csv"
v2_file = "evaluation-VNet2DBTCV-CEL+DICE-DO-nBN.csv"
v3_file = "evaluation-VNet3DBTCV-CEL+DICE-DO-nBN.csv"
en1_file = "evaluation-EqualBTCV-CEL+DICE-DO-nBN.csv"
en4_file = "evaluation-03UNetBTCV-CEL+DICE-DO-nBN.csv"

files = [u2_file, u3_file, v2_file, v3_file, en1_file, en4_file]
names = ["UNet2D", "UNet3D", "VNet2D", "VNet3D", "Ensemble 1", "Ensemble 4"]

metrics = ["Dice - Vein", "Mean Symmetric Surface Distance - Vein", "Confusion Rate - Vein", "Connectivity - Vein",
           "Dice - Artery", "Mean Symmetric Surface Distance - Artery", "Confusion Rate - Artery", "Connectivity - Artery"]

for field in metrics:
    print('-----', field, '-----')
    results = []
    for f in files:
        results.append(pd.read_csv(os.path.join(folder_path, f), usecols=[field]))

    np_results = np.squeeze(np.stack(results))
    means = np.mean(np_results, axis=1)
    if any(x in field for x in ['Dice', 'Connectivity']):
        best = np.argmax(means)
    else:
        best = np.argmin(means)

    print('  Best network is ', names[best], means[best])

    for i in range(len(names)):
        if not i == best:
            t_value, p_value = stats.ttest_rel(np_results[best], np_results[i], axis=0, nan_policy='propagate')
            signi = p_value <= 0.05
            print('  Difference from to', names[i], 'is significant:', signi, '(', p_value, '), mean:', means[i])


    #
    pass