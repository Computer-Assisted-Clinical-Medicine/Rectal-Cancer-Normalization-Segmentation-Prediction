import numpy as np
import SimpleITK as sitk

from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.segbasisloader import SegBasisLoader
from SegmentationNetworkBasis.segratiobasisloader import SegRatioBasisLoader


class VesselSegLoader(SegBasisLoader):

    def adapt_to_task(self, data_img, label_img):
        # threshold_filter = sitk.ThresholdImageFilter()
        # threshold_filter.SetUpper(cfg.num_classes_seg)
        # threshold_filter.SetI
        #
        # (data_img, )
        label_img = sitk.Threshold(label_img, upper=cfg.num_classes_seg-1, outsideValue=cfg.num_classes_seg-1)
        return data_img, label_img

    def _check_images(self, data, lbl):
        # print('To Do: think of check')
        print('          Checking Labels (min, max):', np.min(lbl), np.max(lbl))
        print('          Shapes (Data, Label): ', data.shape, lbl.shape)
        pass


class VesselSegRatioLoader(VesselSegLoader, SegRatioBasisLoader):
    pass
