from scipy import stats
import pandas as pd
import os
import numpy as np

folder_path = "R:\\vessel_results"

u2_file = "evaluation-UNet2D-CEL+DICE-DO-nBN.csv"
u3_file = "evaluation-UNet3D-CEL+DICE-DO-nBN.csv"
v2_file = "evaluation-VNet2D-CEL+DICE-DO-nBN.csv"
v3_file = "evaluation-VNet3D-CEL+DICE-DO-nBN.csv"
d2_file = "evaluation-DVN2D-CEL+DICE-DO-nBN.csv"
d3_file = "evaluation-DVN3D-CEL+DICE-DO-nBN.csv"

files = [u2_file, u3_file, v2_file, v3_file, d2_file, d3_file]
names = ["UNet2D", "UNet3D", "VNet2D", "VNet3D", "DVNet2D", "DVNet3D"]

metrics = ["Dice - Vein", "Connectivity - Vein", "Fragmentation - Vein", "Mean Symmetric Surface Distance - Vein",
           "Dice - Artery", "Connectivity - Artery", "Fragmentation - Artery", "Mean Symmetric Surface Distance - Artery"]

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

    for i in range(6):
        if not i == best:
            t_value, p_value = stats.ttest_rel(np_results[best], np_results[i], axis=0, nan_policy='propagate')
            signi = p_value <= 0.05
            print('  Difference from to', names[i], 'is significant:', signi, '(', p_value, '), mean:', means[i])


    #
    pass