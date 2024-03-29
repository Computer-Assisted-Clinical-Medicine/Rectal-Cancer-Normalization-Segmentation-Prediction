# Segmentation of rectal cancer tumors

This repository runs the training for rectal cancer segmentation and classification using multiple 2D and 3D networks using different normalization methods. Sadly, the training data cannot be made freely available due to data protection. But you are welcome to cooperate with our research group if you want access to the data.

If you use our code in your work please cite the following paper:

Albert, S.; Wichtmann, B.D.; Zhao, W.; Maurer, A.; Hesser, J.; Attenberger, U.I.; Schad, L.R.; Zöllner, F.G. Comparison of Image Normalization Methods for Multi-Site Deep Learning. Appl. Sci. 2023, 13, 8923. https://doi.org/10.3390/app13158923

More details can be found in my dissertation

Albert, Steffen. Prediction of treatment response and outcome in locally advanced rectal cancer using radiomics. Diss. 2023. https://doi.org/10.11588/heidok.00034188

## Getting Started

How everything can be run can be seen in the run.py file. Test data can also be created using the create_test_files.py script, these are also used during the testing.

To try something out, different experiments can be created, which can be trained and evaluated using the Experiment class.

## Prerequisites

- The conda environment is described in the requirements.txt.
- The environmental variable "data_dir" should point to the data directory. train_IDs.csv should contain the test files, dataset.json additional description. Prepare Data Rectal cancer contains classes to write different Dataset formats.
- The variable "experiment_dir" should point to the directory, where the output is saved. The variables can be set in the IDE and are to make everything machine independent.

## Installing

It is best to use virtualenv to create a virtual environment

There is a conflict in the numpy version between tensorflow and the version needed for slicing, so numpy has to be installed with:
python -m pip install --no-warn-conflicts numpy==1.20.0

## Running the tests

- The test can be run using pytest and will create a test_data directory, where the created test data will be saved.
- time_seg_data_loader can be used to identify bottlenecks and profiles the different steps in the loader. For the profiles, snakeviz is used, the command line arguments for visualization are printed at the end of the script. Iti s best to run it in an interactive window for better overview

## Running the training

For training, just execute the run.py. You can use the command it will print ot start tensorboard to monitor the training. Different metric can be analyzed during training. It also supports the profiler and the Graph. HParams is also implemented but does not work very well, even though the normal hook is used.
