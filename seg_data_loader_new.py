import logging
import os

import numpy as np
import SimpleITK as sitk

from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.NetworkBasis.dataloader import DataLoader
from seg_data_loader import SegLoader

#TODO: add data augmentation for new loader
#TODO: introduce optional shift of center

#configure logger
logger = logging.getLogger(__name__)

class SegRatioLoader(DataLoader):

    def adapt_to_task(self, data_img, label_img):
        # threshold_filter = sitk.ThresholdImageFilter()
        # threshold_filter.SetUpper(cfg.num_classes_seg)
        # threshold_filter.SetI
        #
        # (data_img, )
        label_img = sitk.Threshold(label_img, upper=cfg.num_classes_seg-1, outsideValue=cfg.num_classes_seg-1)
        # label should be uint-8
        if label_img.GetPixelID() != sitk.sitkUInt8:
            label_img = sitk.Cast(label_img, sitk.sitkUInt8)
        return data_img, label_img

    def _set_up_shapes_and_types(self):
        """!
        sets all important configurations from the config file:
        - n_channels
        - dtypes
        - dshapes
        - slice_shift

        also derives:
        - data_rank
        - slice_shift

        """
        self.n_channels = cfg.num_channels
        if self.mode is self.MODES.TRAIN or self.mode is self.MODES.VALIDATE:
            self.dtypes = [cfg.dtype, cfg.dtype]
            self.dshapes = [np.array(cfg.train_input_shape), np.array(cfg.train_label_shape)]
            # use the same shape for image and labels
            assert np.all(self.dshapes[0] == self.dshapes[1])
        else:
            self.dtypes = [cfg.dtype]
            self.dshapes = [np.array(cfg.test_data_shape)]

        self.data_rank = len(self.dshapes[0])

    def _set_up_capacities(self):
        """!
        sets buffer size for sample buffer based on cfg.batch_capacity_train and
        cfg.batch_capacity_train based on self.mode

        """
        if self.mode is self.MODES.TRAIN:
            self.sample_buffer_size = cfg.batch_capacity_train
        elif self.mode is self.MODES.VALIDATE:
            self.sample_buffer_size = cfg.batch_capacity_train

    def _get_filenames(self, file_id):
        """For compability reasons, get filenames without the preprocessed ones

        Parameters
        ----------
        file_id : str
            The file id

        Returns
        -------
        [type]
            [description]
        """
        s, _, l, _ = self._get_filenames_cached(file_id)
        return s, l

    def _get_filenames_cached(self, file_id):
        """Gets the filenames and the preprocessed filenames

        Parameters
        ----------
        file_id : str
            The file ID

        Returns
        -------
        str, str, str, str
            The path to the data_file and label_file in the cached and uncached version
        """
        # get the folder and id
        folder, file_number = os.path.split(file_id)
        # generate the file name for the sample
        sample_name = cfg.sample_file_name_prefix + file_number
        data_file = os.path.join(folder, sample_name + cfg.file_suffix)
        if not os.path.exists(data_file):
            raise Exception(f'The file {data_file} could not be found')
        # generate the name of the preprocessed file
        filename = f'{sample_name}_{cfg.normalizing_method}.npy'
        data_file_pre = os.path.join(cfg.preprocessed_dir, filename)

        # generate the file name for the sample
        label_name = cfg.label_file_name_prefix + file_number
        label_file = os.path.join(folder, (label_name + cfg.file_suffix))
        if not os.path.exists(data_file):
            raise Exception(f'The file {label_file} could not be found')
        # generate the name of the preprocessed file
        label_file_pre = os.path.join(cfg.preprocessed_dir, (label_name + '.npy'))
        return data_file, data_file_pre, label_file, label_file_pre

    def _load_file(self, file_name:bytes):
        # convert to string
        file_id = str(file_name, 'utf-8')
        logger.debug('        Loading %s (%s)', file_id, self.mode)
        # Use a SimpleITK reader to load the nii images and labels for training
        data_file, data_file_pre, label_file, label_file_pre = self._get_filenames_cached(file_id) 
        # see if the preprocessed files exist
        if os.path.exists(data_file_pre) and os.path.exists(label_file_pre):
            # load numpy files
            data = np.load(data_file_pre)
            lbl = np.load(label_file_pre)
        else:
            # load images
            data_img = sitk.ReadImage(data_file)
            label_img = sitk.ReadImage(label_file)
            # adapt, resample and normalize them
            data_img, label_img = self.adapt_to_task(data_img, label_img)
            if cfg.do_resampling:
                data_img, label_img = self._resample(data_img, label_img)
            data = sitk.GetArrayFromImage(data_img)
            data = self.normalize(data)
            lbl = sitk.GetArrayFromImage(label_img)
            # convert to right type
            data = data.astype(cfg.dtype_np)
            lbl = lbl.astype(np.uint8)
            # then check them
            self._check_images(data, lbl)
            # if everything is ok, save the preprocessed ones
            np.save(data_file_pre, data)
            np.save(label_file_pre, lbl)

        return data, lbl

    @staticmethod
    def normalize(img, eps=np.finfo(np.float).min):
        img_no_nan = np.nan_to_num(img, nan=cfg.data_background_value)
        # clip outliers and rescale to between zero and one
        a_min = np.quantile(img_no_nan, cfg.norm_min_q)
        a_max = np.quantile(img_no_nan, cfg.norm_max_q)
        if cfg.normalizing_method == cfg.NORMALIZING.PERCENT5:
            img = np.clip(img_no_nan, a_min=a_min, a_max=a_max)
            img = (img - a_min) / (a_max - a_min)
            img = (img * 2) - 1
        elif cfg.normalizing_method == cfg.NORMALIZING.MEAN_STD:
            img = np.clip(img_no_nan, a_min=a_min, a_max=a_max)
            img = img - np.mean(img)
            std = np.std(img)
            img = img / (std if std != 0 else eps)
        elif cfg.normalizing_method == cfg.NORMALIZING.WINDOW:
            img = np.clip(img_no_nan, a_min=cfg.norm_min_v, a_max=cfg.norm_max_v)
            img = (img - cfg.norm_min_v) / (cfg.norm_max_v - cfg.norm_min_v + cfg.norm_eps)
            img = (img * 2) - 1
        else:
            raise NotImplementedError(f'{cfg.normalizing_method} is not implemented')

        return img

    def _check_images(self, data, lbl):
        assert np.any(np.isnan(data)) == False, 'Nans in the image'
        assert np.any(np.isnan(lbl)) == False, 'Nans in the labels'
        assert np.sum(lbl) > 100, 'Not enough labels in the image'
        logger.debug('          Checking Labels (min, max) %s %s:', np.min(lbl), np.max(lbl))
        logger.debug('          Shapes (Data, Label): %s %s', data.shape, lbl.shape)

    def _read_file_and_return_numpy_samples(self, file_name_queue:bytes):
        data, lbl = self._load_file(file_name_queue)
        samples, labels = self._get_samples_from_volume(data, lbl)
        if self.mode is not self.MODES.APPLY:
            return samples, labels
        else:
            return [samples]
        

    def _get_samples_from_volume(self, data, lbl):
        # get the fraction of the samples that should contain the object
        frac_obj = cfg.percent_of_object_samples / 100
        # check that there are labels
        assert np.any(lbl != 0), 'no labels found'
        # check shape
        assert np.all(data.shape[:-1] == lbl.shape)
        assert len(data.shape) == 4, 'data should be 4d'
        assert len(lbl.shape) == 3, 'labels should be 3d'
        
        # determine the number of background and foreground samples
        n_foreground = int(cfg.samples_per_volume * frac_obj)
        n_background = int(cfg.samples_per_volume * (1 - frac_obj))

        # calculate the maximum padding, so that at least half in each dimension is inside the image
        # sample shape is without the number of channels
        if self.data_rank == 4:
            sample_shape = self.dshapes[0][:-1]
        # if the rank is three, add a dimension for the z-extent
        elif self.data_rank == 3:
            sample_shape = np.array([1,]+list(self.dshapes[0][:2]))
        assert sample_shape.size == len(data.shape)-1, 'sample shape does not match data shape'
        max_padding = sample_shape // 4

        # pad the data (using 0s)
        pad_with = ((max_padding[0],)*2, (max_padding[1],)*2, (max_padding[2],)*2)
        data_padded = np.pad(data, pad_with + ((0, 0),))
        label_padded = np.pad(lbl, pad_with)

        # calculate the allowed indices
        # the indices are applied to the padded data, so the minimum is 0
        # the last dimension, which is the number of channels is ignored
        min_index = np.zeros(3, dtype=int)
        # the maximum is the new data shape minus the sample shape (accounting for the padding)
        max_index = data_padded.shape[:-1] - sample_shape 
        assert np.all(min_index <= max_index), 'image to small to get patches'

        # get the background origins
        background_shape = (n_background, 3)
        origins_background = np.random.randint(low=min_index, high=max_index, size=background_shape)

        # get the foreground center
        valid_centers = np.argwhere(lbl)
        indices = np.random.randint(low=0, high=valid_centers.shape[0], size=n_foreground)
        origins_foreground = valid_centers[indices] + max_padding - sample_shape // 2
        # check that they are below the maximum amount of padding
        for i, m in enumerate(max_index):
            origins_foreground[:,i] = np.clip(origins_foreground[:,i], 0, m)

        # extract patches (pad if necessary), in separate function, do augmentation beforehand or with patches
        origins = np.concatenate([origins_foreground, origins_background])
        batch_shape = (n_foreground+n_background,) + tuple(sample_shape)
        samples = np.zeros(batch_shape + (self.n_channels,), dtype=cfg.dtype_np)
        labels = np.zeros(batch_shape, dtype=np.uint8)
        for num, (i,j,k) in enumerate(origins):
            sample_patch = data_padded[i:i+sample_shape[0],j:j+sample_shape[1],k:k+sample_shape[2]]
            label_patch = label_padded[i:i+sample_shape[0],j:j+sample_shape[1],k:k+sample_shape[2]]
            samples[num] = sample_patch
            labels[num] = label_patch
            # if num < n_foreground: # only for debugging
            #     assert np.sum(label_patch) > 0

        if self.mode == self.MODES.APPLY:
            raise NotImplementedError('Use the original data loader')
            # return np.zeros([cfg.batch_size_test] + cfg.test_data_shape), None

        # if rank is 3, squash the z-axes
        if self.data_rank == 3:
            samples = samples.squeeze(axis=1)
            labels = labels.squeeze(axis=1)

        # assert np.sum(labels) > 0 # only for debugging

        # convert to one_hot_label
        if self.mode is not self.MODES.APPLY:
            labels_onehot = np.squeeze(np.eye(cfg.num_classes_seg)[labels.flat]).reshape(labels.shape + (-1,))

        # assert np.sum(labels_onehot[...,1]) > 0 # only for debugging
        # if self.data_rank == 3:
        #     print(np.sum(labels_onehot[...,1], axis=(1,2)).astype(int))
        # else:
        #     print(np.sum(labels_onehot[...,1], axis=(1,2,3)).astype(int))

        return samples, labels_onehot