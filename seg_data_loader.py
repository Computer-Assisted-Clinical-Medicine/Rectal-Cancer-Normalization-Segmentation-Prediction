import logging
from typing import Tuple

import numpy as np
import SimpleITK as sitk

from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.segbasisloader import ApplyBasisLoader, SegBasisLoader

#configure logger
logger = logging.getLogger(__name__)

class SegLoader(SegBasisLoader):

    def adapt_to_task(self, data_img:sitk.Image, label_img:sitk.Image):
        """Adapt the data to the current task, for example by changing which
        labels are included

        Parameters
        ----------
        data_img : sitk.Image
            The data image
        label_img : sitk.Image
            The label image

        Returns
        -------
        sitk.Image, sitk.Image
            The converted images
        """
        label_img = sitk.Threshold(label_img, upper=cfg.num_classes_seg-1, outsideValue=cfg.num_classes_seg-1)
        # label should be uint-8
        if label_img.GetPixelID() != sitk.sitkUInt8:
            label_img = sitk.Cast(label_img, sitk.sitkUInt8)
        return data_img, label_img

class ApplyLoader(ApplyBasisLoader):

    def adapt_to_task(self, data_img, label_img=None):
        if label_img is not None:
            label_img = sitk.Threshold(label_img, upper=cfg.num_classes_seg-1, outsideValue=cfg.num_classes_seg-1)
            # label should be uint-8
            if label_img.GetPixelID() != sitk.sitkUInt8:
                label_img = sitk.Cast(label_img, sitk.sitkUInt8)
        return data_img, label_img

    def _check_images(self, data, lbl):
        return