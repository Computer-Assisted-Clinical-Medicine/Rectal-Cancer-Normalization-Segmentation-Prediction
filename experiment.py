import copy
import logging
import os
import stat
import sys
from pathlib import Path, PurePath
from typing import List

import GPUtil
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tqdm import tqdm

import evaluation
from seg_data_loader import ApplyLoader, SegLoader
from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis import postprocessing
from SegmentationNetworkBasis.NetworkBasis.util import write_configurations

#configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Experiment():

    def __init__(self, name:str, hyper_parameters:dict, data_set:List, external_test_set=None, folds=5, seed=42, num_channels=1,
                 output_path_rel=None, restart=False, reinitialize_folds=False, folds_dir_rel=None, preprocessed_dir_rel=None,
                 tensorboard_images=False):
        """Run experiments using a fixed set of hyperparameters

        Parameters
        ----------
            name : str
                Name of the experiment, is used for the folder name
            hyper_parameters : dict
                the hyperparameters that should be used (as soon as something is changed in between experiments, it is a hyperparameter)
            data_set : List
                The list of images which should be used for training, validation and test
            external_test_set : List, optional
                The list of images if an external test set should also be used
            folds : int, optional
                The number of folds to use for validation, by default 5
            seed : int, optional
                the global seed, by default 42
            num_channels: int, optional
                the number of channels in the data, default 1
            output_path_rel : str, optional
                path to write output in (relative to the experiment_dir env. variable),
                if None Experiments is used, by default None
            restart : bool, optional
                If already finished folds should be restarted, by default False
            reinitialize_folds : bool, optional
                If set to true, the split for the folds will be redone, by default False
            folds_dir_rel : str, optional
                Where the fold descripions should be saved (relative to the experiment_dir env. variable).
                All experiments sharing the same folds should have the same directory here, by default outputdir/folds
            preprocessed_dir_rel : str, optional
                Where the preprocessed files are saved (relative to the experiment_dir env. variable),
            self.tensorboard_images : bool, optional
                Wether to write images to tensorboard, this takes a bit, so should only be used for debugging, by default False
        """
        # do a deep copy of the parameters, because they contain lists and dicts
        self.hyper_parameters = copy.deepcopy(hyper_parameters)
        self.seed = seed
        self.name = name
        self.folds = folds
        self.num_channels = num_channels
        self.reinitialize_folds = reinitialize_folds
        self.data_set = np.array(data_set)
        self.external_test_set = external_test_set
        if self.external_test_set is not None:
            self.external_test_set = np.array(self.external_test_set)

        # get the environmental variables
        self.data_dir = Path(os.environ['data_dir'])
        self.experiment_dir = Path(os.environ['experiment_dir'])

        # check input
        if len(data_set) == 0:
            raise ValueError('Dataset is empty.')
        if cfg.number_of_vald*self.folds > self.data_set.size:
            raise ValueError('Dataset to small for the specified folds.')

        if output_path_rel == None:
            self.output_path_rel = PurePath('Experiments', self.name)
        else:
            self.output_path_rel = PurePath(output_path_rel)
            if self.output_path_rel.is_absolute():
                raise ValueError(f'output_path_rel is an absolute path')

        # set the absolute path (which will not be exported)
        self.output_path = self.experiment_dir / Path(self.output_path_rel)

        if not self.output_path.exists():
            self.output_path.mkdir()
        logger.info('Set %s as output folder, all output will be there', self.output_path)

        #check for finetuning
        if not hasattr(self.hyper_parameters, 'evaluate_on_finetuned'):
            self.hyper_parameters["evaluate_on_finetuned"]=False

        #set hyperparameterfile to store all hyperparameters
        self.experiment_file = self.output_path / 'parameters.yaml'

        # set directory for folds
        if folds_dir_rel == None:
            self.folds_dir_rel = self.output_path / 'folds'
        else:
            self.folds_dir_rel = PurePath(folds_dir_rel)
            if self.folds_dir_rel.is_absolute():
                raise ValueError(f'folds_dir_rel is an absolute path')

        self.folds_dir = self.experiment_dir / Path(self.folds_dir_rel)

        if not self.folds_dir.exists():
            self.folds_dir.mkdir(parents=True)

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
        self.setup_folds(self.data_set, overwrite=self.reinitialize_folds)

        self.restart = restart

        self.preprocessed_dir_rel = PurePath(preprocessed_dir_rel)

        if self.preprocessed_dir_rel.is_absolute():
            raise ValueError(f'preprocessed_dir_rel is an absolute path')

        self.preprocessed_dir = self.experiment_dir / Path(self.preprocessed_dir_rel)
        if not self.preprocessed_dir.exists():
            self.preprocessed_dir.mkdir()

        self.tensorboard_images = tensorboard_images

        # set postprocessing method
        self.postprocessing_method = postprocessing.keep_big_structures

        #export parameters
        self.export_experiment()

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
            # this orders the indices, so shuffle them again
            remaining_indices=np.random.permutation(remaining_indices)
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
        """This function will set up the shapes in the cfg module so that they
        will run on the current GPU.
        """

        cfg.num_channels = self.num_channels
        cfg.train_dim = 128 # the resolution in plane
        cfg.num_slices_train = 32 # the resolution in z-direction

        # determine batch size
        cfg.batch_size_train = self.estimate_batch_size()
        cfg.batch_size_valid = cfg.batch_size_train

        # set shape according to the dimension
        dim = self.hyper_parameters['dimensions']
        if dim == 2:
            # set shape
            cfg.train_input_shape = [cfg.train_dim, cfg.train_dim, cfg.num_channels]
            cfg.train_label_shape = [cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]

            # set sample numbers
            # there are 10-30 layers per image containing foreground data. Half the
            # samples are taken from the foreground, so take about 64 samples
            # to cover all the foreground pixels at least once on average, but
            cfg.samples_per_volume = 64
            cfg.batch_capacity_train = 4*cfg.samples_per_volume # chosen as multiple of samples per volume
            logger.debug('   Train Shapes: %s (input), %s (labels)', cfg.train_input_shape, cfg.train_label_shape)

        elif dim == 3:
            # set shape
            # if batch size too small, decrease z-extent
            if cfg.batch_size_train < 4:
                cfg.num_slices_train = cfg.num_slices_train // 2
                cfg.batch_size_train = cfg.batch_size_train * 2
                # if still to small, decrease patch extent in plane
                if cfg.batch_size_train < 4:
                    cfg.train_dim = cfg.train_dim // 2
                    cfg.batch_size_train = cfg.batch_size_train * 2
            cfg.train_input_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_channels]
            cfg.train_label_shape = [cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_classes_seg]
            
            # set sample numbers
            # most patches should cover the whole tumore, so a lower sample number
            # can be used
            cfg.samples_per_volume = 8
            cfg.batch_capacity_train = 4*cfg.samples_per_volume # chosen as multiple of samples per volume
            logger.debug('   Train Shapes: %s (input), %s (labels)', cfg.train_input_shape, cfg.train_label_shape)
        
        # set the valid batch size
        cfg.batch_size_valid = cfg.batch_size_train
        # see if the batch size is bigger than the validation set
        if cfg.samples_per_volume * cfg.number_of_vald <= cfg.batch_size_valid:
            cfg.batch_size_valid = cfg.samples_per_volume * cfg.number_of_vald
            

    def estimate_batch_size(self):
        """The batch size estimation is basically trail and error. So far tested
        with 128x128x2 patches in 2D and 128x128x32x2 in 3D, if using different
        values, guesstimate the relation to the memory.

        Returns
        -------
        int
            The recommended batch size
        """
        # set batch size
        # determine GPU memory (in MB)
        if len(tf.test.gpu_device_name()) > 0:
            gpu_number = int(tf.test.gpu_device_name()[-1])
            gpu_memory = int(np.round(GPUtil.getGPUs()[gpu_number].memoryTotal))
        else:
            # if no GPU was found, use 8 GB
            gpu_memory = 1024*1024*8

        a_name = self.hyper_parameters['architecture'].get_name()
        dim = self.hyper_parameters['dimensions']

        if a_name == 'UNet':
            # filters scale after the first filter, so use that for estimation
            first_f = self.hyper_parameters['init_parameters']['n_filters'][0]
            if dim == 2:
                # this was determined by trail and error for 128x128x2 patches
                memory_consumption_guess = 2 * first_f
            elif dim == 3:
                # this was determined by trail and error for 128x128x32x2 patches
                memory_consumption_guess = 64 * first_f
        else:
            raise NotImplementedError('No heuristic implemented for this network.')
        
        # return estimated recommended batch number
        return np.round(gpu_memory // memory_consumption_guess)

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
        training_loader = SegLoader(
            name='training_loader',
            **self.hyper_parameters['data_loader_parameters']
        )
        training_dataset = training_loader(
            train_files,
            batch_size=cfg.batch_size_train,
            n_epochs=self.hyper_parameters['train_parameters']['epochs'],
            read_threads=cfg.train_reader_instances
        )
        validation_dataset = SegLoader(
            mode=SegLoader.MODES.VALIDATE,
            name='validation_loader',
            **self.hyper_parameters['data_loader_parameters']
        )(
            vald_files,
            batch_size=cfg.batch_size_valid,
            read_threads=cfg.vald_reader_instances,
            n_epochs=self.hyper_parameters['train_parameters']['epochs']
        )

        # just use one sample with the foreground class using a random train file
        if self.tensorboard_images:
            visualization_dataset = SegLoader(
                name='visualization',
                frac_obj=1,
                samples_per_volume=1,
                **self.hyper_parameters['data_loader_parameters']
            )(
                [train_files[np.random.randint(len(train_files))]],
                batch_size=1,
                read_threads=1,
                n_epochs=self.hyper_parameters['train_parameters']['epochs']
            )
        else:
            visualization_dataset = None

        # only do a graph for the first fold
        write_graph = (folder_name == 'fold-0')

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
            visualization_dataset = visualization_dataset,
            write_graph=write_graph,
            #add training parameters
            **(self.hyper_parameters["train_parameters"])
        )

        return

    def applying(self, folder_name:str, test_files:List, apply_name='apply'):
        """Apply the trained network to the test files

        Parameters
        ----------
        folder_name : str
            Training output will be in the output path in this subfolder
        test_files : List
            List of test files as string
        apply_name : str, optional
            The subfolder where the evaluated files are stored, by default apply
        """
        tf.keras.backend.clear_session()

        # set preprocessing dir
        cfg.preprocessed_dir = self.preprocessed_dir

        version = 'final'

        testloader = ApplyLoader(
            name='test_loader',
            **self.hyper_parameters['data_loader_parameters']
        )

        net = self.hyper_parameters['architecture'](
            self.hyper_parameters['loss'],
            is_training=False,
            model_path=str(self.output_path/folder_name/'models'/'model-final'),
            **(self.hyper_parameters["init_parameters"])
            )

        logger.info('Started applying %s to test datset.', folder_name)

        apply_path = self.output_path/folder_name/apply_name
        for f in tqdm(test_files, desc=f'{folder_name} (test)', unit='file'):
            f_name = Path(f).name

            # do inference
            result_image = apply_path/f'prediction-{f_name}-{version}{cfg.file_suffix}'
            if not result_image.exists():
                net.apply(testloader, f, apply_path=apply_path)

            # postprocess the image
            postprocessed_image = apply_path/f'prediction-{f_name}-{version}-postprocessed{cfg.file_suffix}'
            if not postprocessed_image.exists():
                self.postprocess(result_image, postprocessed_image)

        tf.keras.backend.clear_session()

    def postprocess(self, unprocessed, processed):
        """Postprocess the label images with the method set as postprocessing
        method.

        Parameters
        ----------
        unprocessed : str
            The path of the unprocessed image
        processed : str
            The path of the processed image
        """
        self.postprocessing_method(unprocessed, processed)

    def evaluate_fold(self, folder_name, test_files, name='test', apply_name='apply', version = 'final'):
        """Evaluate the files generated by the network

        Parameters
        ----------
        folder_name : str
            Training output will be in the output path in this subfolder
        test_files : List
            List of test files as string
        name : str, optional
            The name of the test set, by default test
        apply_name : str, optional
            The subfolder where the evaluated files are stored, by default apply
        version : str, optional
            The version of the results to use, by default final
        """

        logger.info('Start evaluation of %s.', folder_name)

        apply_path = self.output_path/folder_name/apply_name
        if not apply_path.exists():
            raise FileNotFoundError(f'The apply path {apply_path} does not exist.')

        eval_file_path = self.output_path / folder_name / f'evaluation-{folder_name}-{version}_{name}.csv'

        # remember the results
        results = []

        for f in test_files:
            f = Path(f)
            folder = f.parent
            file_number = f.name
            prediction_path = apply_path / f'prediction-{f.name}-{version}{cfg.file_suffix}'
            label_path = folder /  (cfg.label_file_name_prefix + file_number + cfg.file_suffix)
            if not label_path.exists():
                logger.info(f'Label {label_path} does not exists. It will be skipped')
                continue
            try:
                result_metrics = {'File Number' : file_number}

                result_metrics = evaluation.evaluate_segmentation_prediction(
                    result_metrics,
                    str(prediction_path),
                    str(label_path)
                )

                #append result to eval file
                results.append(result_metrics)
                logger.info('        Finished Evaluation for %s', file_number)
            except RuntimeError as err:
                logger.error("    !!! Evaluation of %s failed for %s, %s", folder_name, f.name, err)

        # write evaluation results
        results = pd.DataFrame(results)
        results.set_index('File Number', inplace=True)
        results.to_csv(eval_file_path, sep=';')

        return


    def run_all_folds(self):
        """This is just a wrapper for run_fold and runs it for all folds
        """
        self.hyper_parameters["evaluate_on_finetuned"] = False
        self.set_seed()

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

        tqdm.write(f'Starting with {self.name} {folder_name} (Fold {f+1} of {self.folds})')

        train_files = np.loadtxt(self.datasets[f]['train'], dtype='str', delimiter=',')
        vald_files = np.loadtxt(self.datasets[f]['vald'], dtype='str', delimiter=',')
        test_files = np.loadtxt(self.datasets[f]['test'], dtype='str', delimiter=',')

        # add the path
        train_files = np.array([self.data_dir / t for t in train_files])
        vald_files = np.array([self.data_dir / t for t in vald_files])
        test_files = np.array([self.data_dir / t for t in test_files])

        if not folddir.exists():
            folddir.mkdir()

        cfg.num_files = len(train_files)
        assert cfg.number_of_vald == len(vald_files), 'Wrong number of valid files'

        logger.info(
            '  Data Set %s: %s  train cases, %s  test cases, %s vald cases',
            f, train_files.size, vald_files.size, test_files.size
        )

        self._set_parameters_according_to_dimension()

        epoch_samples = cfg.samples_per_volume * cfg.num_files
        if not epoch_samples % cfg.batch_size_train == 0:
            print('Sample Number not divisible by batch size, epochs will run a little bit short.')
        if cfg.batch_size_train > cfg.samples_per_volume * cfg.num_files:
            print('Reduce batch size to epoch size')
            cfg.batch_size_train = cfg.samples_per_volume * cfg.num_files

        #try the actual training
        model_final = folddir / 'models' / 'model-final'
        if self.restart == False and model_final.exists():
            tqdm.write('Already trained, skip training.')
            logger.info('Already trained, skip training.')
        else:
            self.training(folder_name, train_files, vald_files)

        # do the application and evaluation
        eval_file_path = folddir / f'evaluation-{folder_name}-final_test.csv'
        if eval_file_path.exists():
            tqdm.write('Already evaluated, skip evaluation.')
            logger.info('Already evaluated, skip evaluation.')
        else:
            self.applying(folder_name, test_files)
            self.evaluate_fold(folder_name, test_files)

        # evaluate the postprocessed files
        eval_file_path = folddir / f'evaluation-{folder_name}-final_postprocessed_test.csv'
        if not eval_file_path.exists():
            self.evaluate_fold(folder_name, test_files, version='final-postprocessed')

        # evaluate the external set if present
        if self.external_test_set is not None:
            # add the path
            external_test_set = np.array([self.data_dir / t for t in self.external_test_set])
            ext_eval = folddir / f'evaluation-{folder_name}-final_external_testset.csv'
            if ext_eval.exists():
                tqdm.write('Already evaluated on external set, skip evaluation.')
                logger.info('Already evaluated on external set, skip evaluation.')
            else:
                self.applying(
                    folder_name,
                    external_test_set,
                    apply_name='apply_external_testset'
                )
                self.evaluate_fold(
                    folder_name,
                    external_test_set,
                    name='external_testset',
                    apply_name='apply_external_testset'
                )

            ext_eval = folddir / f'evaluation-{folder_name}-final_postprocessed_external_testset.csv'
            if not ext_eval.exists():
                self.evaluate_fold(
                    folder_name,
                    external_test_set,
                    name='external_testset',
                    apply_name='apply_external_testset',
                    version='final-postprocessed'
                )

        tqdm.write(f'Finished with {self.name} {folder_name} (Fold {f+1} of {self.folds})')

        return

    def evaluate(self, name='test'):
        """Evaluate the training over all folds

        name : str, optional
            The name of the set to evaluate, by default test

        Raises
        ------
        FileNotFoundError
            If and eval file was not found, most likely because the training failed
            or is not finished yet
        """
        for version in ['final', 'final-postprocessed']:
            #set eval files
            eval_files = []
            for f_name in self.fold_dir_names:
                eval_files.append(
                    self.output_path / f_name / f'evaluation-{f_name}-{version}_{name}.csv'
                )
            if not np.all([f.exists() for f in eval_files]):
                print(eval_files)
                raise FileNotFoundError('Eval file not found')
            #combine previous evaluations
            output_path = self.output_path / f'results_{name}_{version}'
            if not output_path.exists():
                output_path.mkdir()
            evaluation.combine_evaluation_results_from_folds(
                output_path,
                eval_files
            )
            #make plots
            evaluation.make_boxplot_graphic(
                output_path,
                output_path / 'evaluation-all-files.csv'
            )

    def evaluate_external_testset(self):
        """evaluate the external testset, this just call evaluate for this set.
        """
        self.evaluate(name='external_testset')

    def export_experiment(self):
        experiment_dict = {
            'name' : self.name,
            'hyper_parameters' : self.hyper_parameters,
            'data_set' : [str(f) for f in self.data_set],
            'folds' : self.folds,
            'seed' : self.seed,
            'num_channels' : self.num_channels,
            'output_path_rel' : self.output_path_rel,
            'restart' : self.restart,
            'reinitialize_folds' : self.reinitialize_folds,
            'folds_dir_rel' : self.folds_dir_rel,
            'preprocessed_dir_rel' : self.preprocessed_dir_rel,
            'tensorboard_images' : self.tensorboard_images
        }
        if self.external_test_set is not None:
            ext_set = [str(f) for f in self.external_test_set]
            experiment_dict['external_test_set'] = ext_set
        with open(self.experiment_file, 'w') as f:
            yaml.dump(experiment_dict, f)

    def from_file(file):
        with open(file, 'r') as f:
            parameters = yaml.load(f, Loader=yaml.Loader)
        return Experiment(**parameters)

    def export_slurm_file(self, working_dir):
        run_script = Path(sys.argv[0]).resolve().parent / 'run_single_experiment.py'
        job_dir = Path(os.environ['experiment_dir']) / 'slurm_jobs'
        if not job_dir.exists():
            job_dir.mkdir()
        log_dir = self.output_path / 'slurm_logs'
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        job_file = job_dir / f'run_{self.name}.sh'
        if self.hyper_parameters['dimensions'] == 3:
            gpu_type = 'GPU_no_K80'
        else:
            gpu_type = 'GPU'
        export_slurm_job(
            filename = job_file,
            command=f'python {run_script} -f $SLURM_ARRAY_TASK_ID -e {self.output_path}',
            job_name=self.name,
            venv_dir=Path(sys.argv[0]).resolve().parent / 'venv',
            workingdir=working_dir,
            job_type=gpu_type,
            hours=24,
            minutes=0,
            log_dir=log_dir,
            array_job=True,
            array_range=f'0-{self.folds-1}',
            variables={
                'data_dir' : os.environ['data_dir'],
                'experiment_dir' : os.environ['experiment_dir']
            }
        )
        return job_file


def export_slurm_job(filename, command, job_name=None, workingdir=None, venv_dir='venv',
    job_type='CPU', cpus=1, hours=0, minutes=30, log_dir=None, log_file=None, error_file=None,
    array_job=False, array_range='0-4', singleton=False, variables={}):
    """Generates a slurm file to run jobs on the cluster

    Parameters
    ----------
    filename : Path or str
        Where the slurm file should be saved
    command : str
        The command to run (can also be multiple commands separated by line breaks)
    job_name : str, optional
        The name displayed in squeue and used for log_name, by default None
    workingdir : str, optional
        The directory in Segmentation_Experiment, if None, basedir is used, by default None
    venv_dir : str, optional
        The directory of the virtual environment, by default venv
    job_type : str, optional
        type of job, CPU, GPU or GPU_no_K80, by default 'CPU'
    cpus : int, optional
        number of CPUs, by default 1
    hours : int, optional
        Time the job should run in hours, by default 0
    minutes : int, optional
        Time the job should run in minutes, by default 30
    log_dir : str, optional
        dir where the logs should be saved if None logs/job_name/, by default None
    log_file : str, optional
        name of the log file, if None job_name_job_id_log.txt, by default None
    error_file : str, optional
        name of the errors file, if None job_name_job_id_log_errors.txt, by default None
    array_job : bool, optional
        If set to true, array_range should be set, by default False
    array_range : str, optional
        array_range as str (comma separated or start-stop (ends included)), by default '0-4'
    singleton : bool, optional
        if only one job with that name and user should be running, by default False
    variables : dict, optional
        environmental variables to write {name : value} $EXPDIR can be used, by default {}
    """
    # this new node dos not work
    exclude_nodes = ['h08c0301', 'h08c0401', 'h08c0501']
    if job_type == 'GPU_no_K80':
        exclude_nodes += [
            'h05c0101', 'h05c0201', 'h05c0301', 'h05c0401', 'h05c0501', 'h06c0301',
            'h05c0601', 'h05c0701', 'h05c0801', 'h05c0901', 'h06c0101', 'h06c0201',
            'h06c0401', 'h06c0501', 'h06c0601', 'h06c0701', 'h06c0801', 'h06c0901'
        ]

    if job_type == 'CPU':
        assert(hours==0)
        assert(minutes <= 30)
    else:
        assert(minutes < 60)
        assert(hours <= 48)

    if log_dir is None:
        log_dir = Path('logs/{job_name}/')
    else:
        log_dir = Path(log_dir)

    if log_file is None:
        if array_job:
            log_file = log_dir / f'{job_name}_%a_%A_log.txt'
        else:
            log_file = log_dir / f'{job_name}_%j_log.txt'
    else:
        log_file = log_dir / log_file

    if error_file is None:
        if array_job:
            error_file = log_dir / f'{job_name}_%a_%A_errors.txt'
        else:
            error_file = log_dir / f'{job_name}_%j_errors.txt'
    else:
        error_file = log_dir / error_file

    filename = Path(filename)

    slurm_file = '#!/bin/bash\n\n'
    if job_name is not None:
        slurm_file += f"#SBATCH --job-name={job_name}\n"

    slurm_file += f'#SBATCH --cpus-per-task={cpus}\n'
    slurm_file += f'#SBATCH --ntasks-per-node=1\n'
    slurm_file += f'#SBATCH --time={hours:02d}:{minutes:02d}:00\n'
    slurm_file += '#SBATCH --mem=32gb\n'

    if job_type == 'GPU' or job_type == 'GPU_no_K80':
        slurm_file += '\n#SBATCH --partition=gpu-single\n'
        slurm_file += '#SBATCH --gres=gpu:1\n'

    if len(exclude_nodes) > 0:
        slurm_file += '#SBATCH --exclude=' + ','.join(exclude_nodes) + '\n'

    if array_job:
        slurm_file += f'\n#SBATCH --array={array_range}\n'

    # add logging
    slurm_file += f'\n#SBATCH --output={str(log_file)}\n'
    slurm_file += f'#SBATCH --error={str(error_file)}\n'

    if singleton:
        slurm_file += '\n#SBATCH --dependency=singleton\n'

    # define workdir, add diagnostic info
    slurm_file += '''
echo "Set Workdir"
WSDIR=/gpfs/bwfor/work/ws/hd_mo173-myws
echo $WSDIR
EXPDIR=$WSDIR\n'''

    # print task ID depending on type
    if array_job:
        slurm_file +='\necho "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID\n'
    else:
        slurm_file +='\necho "My SLURM_JOB_ID: " $SLURM_JOB_ID\n'

    slurm_file += '''\necho "job started on Node: $HOSTNAME"

echo "Load modules"

module load devel/python_intel/3.7
'''

    # add environmental variables
    if len(variables) > 0:
        slurm_file += '\n'
    for key, val in variables.items():
        slurm_file += f'export {key}="{val}"\n'

    if 'GPU' in job_type:
        slurm_file += '''module load devel/cuda/10.1
module load lib/cudnn/7.6.5-cuda-10.1

echo "Get GPU info"
nvidia-smi
'''

    slurm_file += '\necho "Go to workingdir"\n'
    if workingdir is None:
        slurm_file += 'cd $EXPDIR/nnUNet\n'
    else:
        slurm_file += f'cd {Path(workingdir).resolve()}\n'

    # activate virtual environment
    slurm_file += '\necho "Activate virtual environment"\n'
    slurm_file += f'source {Path(venv_dir).resolve()}/bin/activate\n'


    # run the real command
    slurm_file += '\necho "Start calculation"\n\n'
    slurm_file += command
    slurm_file += '\n\necho "Finished"'
    
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)
    # write to file
    with open(filename, 'w+') as f:
        f.write(slurm_file)

    return

def export_batch_file(filename, commands):
    """Exports a list of commands (one per line) as batch script

    Parameters
    ----------
    filename : str or Path
        The new file
    commands : [str]
        List of commands (as strings)
    """

    filename = Path(filename)

    batch_file = '#!/bin/bash'

    for c in commands:
        batch_file += f'\n\n{c}'

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)
    # write to file
    with open(filename, 'w+') as f:
        f.write(batch_file)

    # set permission
    os.chmod(filename, stat.S_IRWXU)

    return
