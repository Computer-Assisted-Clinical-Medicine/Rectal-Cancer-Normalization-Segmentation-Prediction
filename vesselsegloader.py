import numpy as np
from SegmentationNetworkBasis.segbasisloader import SegBasisLoader
from SegmentationNetworkBasis.segratiobasisloader import SegRatioBasisLoader


class VesselSegLoader(SegBasisLoader):

    def adapt_to_task(self, data_img, label_img):
        label_img = label_img > 0  # combine liver and tumor labels
        return data_img, label_img

    def _check_images(self, data, lbl):
        # Dims: H, W, Z
        left = np.sum(lbl[:, :150, :])
        right = np.sum(lbl[:, -150:, :])
        if left > right:
            print('            LIVER IS LEFT. (', left, right, ')')
        else:
            print('            LIVER IS RIGHT. (', left, right, ')')


class VesselSegRatioLoader(VesselSegLoader, SegRatioBasisLoader):
    pass





