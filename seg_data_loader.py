import logging
import os

import numpy as np
import SimpleITK as sitk

from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.segbasisloader import SegBasisLoader
from SegmentationNetworkBasis.segratiobasisloader import SegRatioBasisLoader
from SegmentationNetworkBasis.NetworkBasis import image as Image

#configure logger
logger = logging.getLogger(__name__)

class SegLoader(SegBasisLoader):

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

    def _check_images(self, data, lbl):
        assert np.any(np.isnan(data)) == False, 'Nans in the image'
        assert np.any(np.isnan(lbl)) == False, 'Nans in the labels'
        assert np.sum(lbl) > 100, 'Not enough labels in the image'
        logger.debug('          Checking Labels (min, max) %s %s:', np.min(lbl), np.max(lbl))
        logger.debug('          Shapes (Data, Label): %s %s', data.shape, lbl.shape)

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
        filename = f'{sample_name}_{cfg.normalizing_method}_old_format.npy'
        data_file_pre = os.path.join(cfg.preprocessed_dir, filename)

        # generate the file name for the sample
        label_name = cfg.label_file_name_prefix + file_number
        label_file = os.path.join(folder, (label_name + cfg.file_suffix))
        if not os.path.exists(data_file):
            raise Exception(f'The file {label_file} could not be found')
        # generate the name of the preprocessed file
        label_file_pre = os.path.join(cfg.preprocessed_dir, (label_name + '_old_format.npy'))
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
            # move z axis to last index (do not use -1 in case there are 4 dimensions)
            data = np.moveaxis(data, 0, 2)
            lbl = np.moveaxis(lbl, 0, 2)
            self._check_images(data, lbl)
            self._check_images(data, lbl)
            # if everything is ok, save the preprocessed ones
            np.save(data_file_pre, data)
            np.save(label_file_pre, lbl)

        if self.mode is self.MODES.APPLY:
            data = Image.pad_image(data, 'edge', self.slice_shift)
            lbl = Image.pad_image(lbl, 'constant', self.slice_shift, cfg.label_background_value)

        return data, lbl

class SegRatioLoader(SegLoader, SegRatioBasisLoader):
    pass
