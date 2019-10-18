import pandas as pd
import os.path


for data_set in ['../btcv.csv', '../chaos.csv', '../ircad.csv', '../lits.csv', '../tcia.csv', '../anat3.csv', '../silvcorp.csv']:
    print('------------------', data_set, '------------------')
    all_files = pd.read_csv(data_set, dtype=object).as_matrix()
    for file in all_files:
        folder, file_number = os.path.split(file[0])
        path = os.path.join(folder, ('volume-' + file_number + '.nii'))
        if os.path.isfile(path):
            print(path, 'ok')
        else:
            print(path, 'DOES NOT EXIST!')

