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

    def _get_filenames(self, file_id):
        data_file = os.path.join(file_id, (cfg.sample_file_name))
        label_file = os.path.join(file_id, (cfg.label_file_name))
        return data_file, label_file

    def adapt_to_task(self, data_img, label_img):
        # threshold_filter = sitk.ThresholdImageFilter()
        # threshold_filter.SetUpper(cfg.num_classes_seg)
        # threshold_filter.SetI
        #
        # (data_img, )
        label_img = sitk.Threshold(label_img, upper=cfg.num_classes_seg-1, outsideValue=cfg.num_classes_seg-1)
        return data_img, label_img

    def _check_images(self, data, lbl):
        # print('ToDo: think of check')
        logger.debug('          Checking Labels (min, max) %s %s:', np.min(lbl), np.max(lbl))
        logger.debug('          Shapes (Data, Label): %s %s', data.shape, lbl.shape)


class SegRatioLoader(SegLoader, SegRatioBasisLoader):
    pass
