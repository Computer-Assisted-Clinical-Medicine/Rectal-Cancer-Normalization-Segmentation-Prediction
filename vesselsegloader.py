import numpy as np
from SegmentationNetworkBasis.segbasisloader import SegBasisLoader
from SegmentationNetworkBasis.segratiobasisloader import SegRatioBasisLoader


class VesselSegLoader(SegBasisLoader):

    def adapt_to_task(self, data_img, label_img):
        return data_img, label_img

    def _check_images(self, data, lbl):
        # print('To Do: think of check')
        print('          Checking Labels (min, max):', np.min(lbl), np.max(lbl))
        print('          Shapes (Data, Label): ', data.shape, lbl.shape)
        pass


class VesselSegRatioLoader(VesselSegLoader, SegRatioBasisLoader):
    pass





