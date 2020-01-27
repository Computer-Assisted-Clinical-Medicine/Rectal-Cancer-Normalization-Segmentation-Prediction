import numpy as np
import os
import glob
from SegmentationNetworkBasis import config as cfg

data_sets = [('ircad', 'D:\Image_Data\Patient_Data\IRCAD_new'), ('xcat', 'D:\Image_Data\\Numerical_Phantoms\\XCAT\\NRRD_Breathe'),
             ('gan', 'D:\Image_Data\Patient_Data\IRCAD_new\Synthetic'), ('btcv', 'T:\BTCV'), ('synth', 'T:\DVN-Synth')]

for (data_name, data_path) in data_sets:
    print('Data Set: ', data_name)
    files = []
    for case in glob.glob(os.path.join(data_path, 'Data', cfg.label_file_name_prefix + '*.nii')):
        path, file = os.path.split(case)
        file = os.path.splitext(file)[0]
        _, _, number = file.split('-')
        entry = os.path.join(data_path, 'Data', number)
        print('   ' + file)
        files.append(entry)
    np.savetxt('..\\' + data_name + '.csv', files, fmt='%s', header='path')