import logging
import os

import numpy as np
import SimpleITK as sitk

from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.segbasisloader import SegBasisLoader
from SegmentationNetworkBasis.segratiobasisloader import SegRatioBasisLoader

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

class SegRatioLoader(SegLoader, SegRatioBasisLoader):
    pass
