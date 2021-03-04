import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

import evaluation
from seg_data_loader import ApplyLoader, SegLoader
from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.NetworkBasis.util import (make_csv_file,
                                                        write_configurations,
                                                        write_metrics_to_csv)

#configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Experiment():

    def __init__(self, name:str, hyper_parameters:dict, data_set:List, folds=5, seed=42, num_channels=1,
                 output_path=None, restart=False, reinitialize_folds=False, folds_dir=None, preprocessed_dir=None):
        """Run experiments using a fixed set of hyperparameters

        Parameters
        ----------
            name : str
                Name of the experiment, is used for the folder name
            hyper_parameters : dict
                the hyperparameters that should be used (as soon as something is changed in between experiments, it is a hyperparameter)
            data_set : List
                The list of images which should be used for training, validation and test
            folds : int
                The number of folds to use for validation, by default 5
            seed : int, optional
                the global seed, by default 42
            num_channels: int, optional
                the number of channels in the data, default 1
            output_path : str, optional
                path to write output in, if None and not on server, Experiments is used, by default None
            restart : bool, optional
                If already finished folds should be restarted, by default False
            reinitialize_folds : bool, optional
                If set to true, the split for the folds will be redone, by default False
            folds_dir : str, optional
                Where the fold descripions should be saved. All experiments sharing the 
                same folds should have the same directory here, by default outputdir
            preprocessed_dir : str, optional
                Where the preprocessed files are saved
        """
        self.hyper_parameters = hyper_parameters
        self.seed = seed
        self.name = name
        self.folds = folds
        self.num_channels = num_channels

        #set path different on the Server
        if cfg.ONSERVER:
            self.output_path = Path("tmp", self.name)
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            if output_path == None:
                self.output_path = Path('Experiments', self.name)
            else:
                self.output_path = Path(output_path)

        if not self.output_path.exists():
            self.output_path.mkdir()
        logger.info('Set %s as output folder, all output will be there', self.output_path)

        #check for finetuning
        if not hasattr(self.hyper_parameters, 'evaluate_on_finetuned'):
            self.hyper_parameters["evaluate_on_finetuned"]=False

        #set hyperparameterfile to store all hyperparameters
        self.hyperparameter_file = self.output_path / 'hyperparameters.json'

        # set directory for folds
        if folds_dir == None:
            self.folds_dir = self.output_path
        else:
            self.folds_dir = Path(folds_dir)
        if not folds_dir.exists():
            folds_dir.mkdir(parents=True)

        #set fold directory names
        self.fold_dir_names = [f'fold-{f}' for f in range(self.folds)]
        #set fold split file names
        self.datasets = []
        for f in range(self.folds):
            # set paths
            train_csv = self.folds_dir/f'train-{f}-{self.folds}.csv'
            vald_csv = self.folds_dir/f'vald-{f}-{self.folds}.csv'
            test_csv = self.folds_dir/f'test-{f}-{self.folds}.csv'
            self.datasets.append({
                'train' : train_csv,
                'vald' : vald_csv,
                'test' : test_csv
            })
        # to the data split
        self.setup_folds(data_set, overwrite=reinitialize_folds)
        self.data_set = data_set

        self.restart = restart

        self.preprocessed_dir = preprocessed_dir

        return

    def set_seed(self):
        """Set the seed in tensorflow and numpy
        """
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def setup_folds(self, data_set:List, overwrite=False):
        """Setup the split of the dataset. This will be done in the output_path
        and can be used by all experiments in that path.

        Parameters
        ----------
        data_set : List
            The files in the dataset as list
        overwrite : bool, optional
            IF this is true, existing files are overwritten, by default False
        """
        self.set_seed()

        all_indices = np.random.permutation(range(0, data_set.size))
        #split the data into self.folds sections
        if self.folds > 1:
            test_folds = np.array_split(all_indices, self.folds)
        else:
            #otherwise, us cfg.data_train_split
            test_folds = all_indices[int(all_indices.size*cfg.data_train_split):].reshape(1,-1)

        for f in range(0, self.folds):
            #test is the section
            test_indices = test_folds[f]
            remaining_indices = np.setdiff1d(all_indices, test_folds[f])
            #number of validation is set in config
            vald_indices = remaining_indices[:cfg.number_of_vald]
            #the rest is used for training
            train_indices = remaining_indices[cfg.number_of_vald:]

            train_files = data_set[train_indices]
            vald_files = data_set[vald_indices]
            test_files = data_set[test_indices]

            # only write files if they do not exist or overwrite is true
            if not self.datasets[f]['train'].exists() or overwrite:
                np.savetxt(self.datasets[f]['train'], train_files, fmt='%s', header='path')
            if not self.datasets[f]['vald'].exists() or overwrite:
                np.savetxt(self.datasets[f]['vald'], vald_files, fmt='%s', header='path')
            if not self.datasets[f]['test'].exists() or overwrite:
                np.savetxt(self.datasets[f]['test'], test_files, fmt='%s', header='path')
        return

    def _set_parameters_according_to_dimension(self):
        """This function will set up the shapes in the cfg module
        """
        if self.hyper_parameters['dimensions'] == 2:
            cfg.num_channels = self.num_channels
            cfg.train_dim = 128
            cfg.samples_per_volume = 128
            cfg.batch_capacity_train = 4*cfg.samples_per_volume # chosen as multiple of samples per volume
            cfg.train_input_shape = [cfg.train_dim, cfg.train_dim, cfg.num_channels]
            cfg.train_label_shape = [cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
            logger.debug('   Train Shapes: %s (input), %s (labels)', cfg.train_input_shape, cfg.train_label_shape)
            cfg.test_dim = 256
            cfg.test_data_shape = [cfg.test_dim, cfg.test_dim, cfg.num_channels]
            cfg.test_label_shape = [cfg.test_dim, cfg.test_dim, cfg.num_classes_seg]
            logger.debug('   Test Shapes: %s (input) %s (labels)', cfg.test_data_shape, cfg.test_label_shape)
            # set batch size
            cfg.batch_size_train = 256
            # use smaller batch size for some networks
            if self.hyper_parameters['architecture'].get_name() == 'ResNet':
                cfg.batch_size_train = 16
        elif self.hyper_parameters['dimensions'] == 3:
            cfg.num_channels = self.num_channels
            cfg.train_dim = 128
            cfg.samples_per_volume = 32
            cfg.batch_capacity_train = 4*cfg.samples_per_volume # chosen as multiple of samples per volume
            cfg.num_slices_train = 32 # with 16 the old loader does not work, but with 8, the U-Net does not work.
            cfg.train_input_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_channels]
            cfg.train_label_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
            logger.debug('   Train Shapes: %s (input), %s (labels)', cfg.train_input_shape, cfg.train_label_shape)
            cfg.test_dim = 256
            cfg.num_slices_test = 32
            cfg.test_data_shape = [cfg.num_slices_test, cfg.test_dim, cfg.test_dim, cfg.num_channels]
            cfg.test_label_shape = [cfg.num_slices_test, cfg.test_dim, cfg.test_dim, cfg.num_classes_seg]
            logger.debug('   Test Shapes: %s (input) %s (labels)', cfg.test_data_shape, cfg.test_label_shape)
            cfg.batch_size_train = 8
            if self.hyper_parameters['architecture'].get_name() == 'VNet':
                cfg.batch_size_train = 4 # Otherwise, VNet 3D fails
            elif self.hyper_parameters['architecture'].get_name() == 'ResNet': # Does not work, not enough memory on PC
                cfg.batch_size_train = 2
                cfg.train_dim = 32
                cfg.num_slices_train = 16
                cfg.train_input_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_channels]
                cfg.train_label_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]

    def training(self, folder_name:str, train_files:List, vald_files:List):
        """Do the actual training

        Parameters
        ----------
        folder_name : str
            Training output will be in the output path in this subfolder
        train_files : List
            List of training files as string
        vald_files : List
            List of validation files as string
        """
        tf.keras.backend.clear_session()

        # set preprocessing dir
        cfg.preprocessed_dir = self.preprocessed_dir

        #generate loader
        training_loader = SegLoader(name='training_loader')
        training_dataset = training_loader(
            train_files,
            batch_size=cfg.batch_size_train,
            n_epochs=self.hyper_parameters['train_parameters']['epochs'],
            read_threads=cfg.train_reader_instances
        )
        validation_dataset = SegLoader(
            mode=SegLoader.MODES.VALIDATE,
            name='validation_loader'
        )(
            vald_files,
            batch_size=cfg.batch_size_train,
            read_threads=cfg.vald_reader_instances,
            n_epochs=self.hyper_parameters['train_parameters']['epochs'],
        )

        net = self.hyper_parameters['architecture'](
            self.hyper_parameters['loss'],
            #add initialization parameters
            **self.hyper_parameters["init_parameters"]
        )
        write_configurations(self.output_path, folder_name, net, cfg)
        # Train the network with the dataset iterators
        logger.info('Started training of %s', folder_name)
        net.train(
            logs_path=str(self.output_path),
            folder_name=folder_name,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            #add training parameters
            **(self.hyper_parameters["train_parameters"])
        )

        return

    def applying(self, folder_name:str, test_files:List):
        """Apply the trained network to the test files

        Parameters
        ----------
        folder_name : str
            Training output will be in the output path in this subfolder
        test_files : List
            List of test files as string
        """
        tf.keras.backend.clear_session()

        # set preprocessing dir
        cfg.preprocessed_dir = self.preprocessed_dir

        testloader = ApplyLoader(
            name='test_loader'
        )

        net = self.hyper_parameters['architecture'](
            self.hyper_parameters['loss'],
            is_training=False,
            model_path=str(self.output_path/folder_name/'models'/'model-final'),
            **(self.hyper_parameters["init_parameters"])
            )

        logger.info('Started applying %s to test datset.', folder_name)
        for f in tqdm(test_files, desc=f'{folder_name} (test)', unit='file'):
            net.apply(testloader, f, apply_path=self.output_path/folder_name/'apply')

        tf.keras.backend.clear_session()

    def evaluate_fold(self, folder_name, test_files):
        '''!
        do testing

        '''

        logger.info('Start evaluation of %s.', folder_name)

        apply_path = self.output_path/folder_name/'apply'
        if not apply_path.exists():
            apply_path.mkdir()

        version = 'final'

        eval_file_path = self.output_path / folder_name / f'evaluation-{folder_name}-{version}_test.csv'
        header_row = evaluation.make_csv_header()
        make_csv_file(eval_file_path, header_row)

        for f in test_files:
            f = Path(f)
            folder = f.parent
            file_number = f.name
            prediction_path = Path(apply_path) / f'prediction-{f.name}-{version}{cfg.file_suffix}'

            label_path = folder /  (cfg.label_file_name_prefix + file_number + cfg.file_suffix)
            try:
                result_metrics = {}
                result_metrics['File Number'] = file_number

                result_metrics = evaluation.evaluate_segmentation_prediction(result_metrics, str(prediction_path), str(label_path))

                #append result to eval file
                write_metrics_to_csv(eval_file_path, header_row, result_metrics)
                logger.info('        Finished Evaluation for %s', file_number)
            except RuntimeError as err:
                logger.error("    !!! Evaluation of %s failed for %s, %s", folder_name, f.name, err)
        
        return


    def run_all_folds(self):
        """This is just a wrapper for run_fold and runs it for all folds
        """
        self.hyper_parameters["evaluate_on_finetuned"] = False
        self.set_seed()

        #export parameters
        self.export_hyperparameters()

        for f, in range(0, self.folds):
            self.run_fold(f)

        return

    def run_fold(self, f:int):
        """Run the training and evaluation for all folds

        Parameters
        ----------
        f : int
            The number of the fold
        """

        folder_name = self.fold_dir_names[f]
        folddir = self.output_path / folder_name
        logger.info('workingdir is %s', folddir)

        # skip already finished folds
        if self.restart == False:
            eval_file_path = folddir / f'evaluation-{folder_name}-final_test.csv'
            if eval_file_path.exists():
                tqdm.write('Already trained, skip to next fold')
                logger.info('Already trained, skip to next fold')
                return

        train_files = np.loadtxt(self.datasets[f]['train'], dtype='str', delimiter=',')
        vald_files = np.loadtxt(self.datasets[f]['vald'], dtype='str', delimiter=',')
        test_files = np.loadtxt(self.datasets[f]['test'], dtype='str', delimiter=',')

        if not folddir.exists():
            folddir.mkdir()

        cfg.num_files = len(train_files)
        cfg.num_files_vald = len(vald_files)

        logger.info(
            '  Data Set %s: %s  train cases, %s  test cases, %s vald cases',
            f, train_files.size, vald_files.size, test_files.size
        )

        self._set_parameters_according_to_dimension()

        tqdm.write(f'Starting with {self.name} {folder_name} (Fold {f+1} of {self.folds})')

        epoch_samples = cfg.samples_per_volume * cfg.num_files
        if not epoch_samples % cfg.batch_size_train == 0:
            print('Sample Number not divisible by batch size, consider changing it.')

        # set the trial id for tensorboard
        cfg.trial_id = f'{self.name} - fold {f}'

        #try the actual training
        cfg.percent_of_object_samples = 50
        self.training(folder_name, train_files, vald_files)

        self.applying(folder_name, test_files)

        self.evaluate_fold(folder_name, test_files)

        tqdm.write(f'Finished with {self.name} {folder_name} (Fold {f+1} of {self.folds})')

        return

    def evaluate(self):
        """Evaluate the training over all folds

        Raises
        ------
        FileNotFoundError
            If and eval file was not found, most likely because the training failed
            or is not finished yet
        """
        #set eval files
        eval_files = []
        epochs = str(self.hyper_parameters['train_parameters']['epochs'])
        for f_name in self.fold_dir_names:
            eval_files.append(
                self.output_path / f_name / f'evaluation-{f_name}-{epochs}_test.csv'
            )
        if not np.all([f.exists() for f in eval_files]):
            print(eval_files)
            raise FileNotFoundError('Eval file not found')
        #combine previous evaluations
        evaluation.combine_evaluation_results_from_folds(
            self.output_path,
            eval_files
        )
        #make plots
        evaluation.make_boxplot_graphic(
            self.output_path,
            eval_files
        )

    def export_hyperparameters(self):
        with open(self.hyperparameter_file, 'w') as f:
            yaml.dump(self.hyper_parameters, f)
