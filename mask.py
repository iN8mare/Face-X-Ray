import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Mask:
    def __init__(self, landmarks, face, channels=4):
        logger.info("Initializing %s: (face_shape: %s, channels: %s, landmarks: %s)",
                     self.__class__.__name__, face.shape, channels, landmarks)
        self.landmarks = landmarks
        self.face = face
        self.channels = channels

        mask = self.build_mask()
        self.mask = self.merge_mask(mask)
        logger.info("Initialized %s", self.__class__.__name__)

    def merge_mask(self, mask):
        """ Return the mask in requested shape """
        logger.info("mask_shape: %s", mask.shape)
        assert self.channels in (1, 3, 4), "Channels should be 1, 3 or 4"
        assert mask.shape[2] == 1 and mask.ndim == 3, "Input mask be 3 dimensions with 1 channel"

        if self.channels == 3:
            retval = np.tile(mask, 3)
        elif self.channels == 4:
            retval = np.concatenate((self.face, mask), -1)
        else:
            retval = mask

        logger.info("Final mask shape: %s", retval.shape)
        return retval

class facehull(Mask):  # pylint: disable=invalid-name
    """ Basic face hull mask """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)
        hull = cv2.convexHull(  # pylint: disable=no-member
            np.array(self.landmarks).reshape((-1, 2)))
        cv2.fillConvexPoly(mask, hull, 255.0, lineType=cv2.LINE_AA)  # pylint: disable=no-member
        return mask
