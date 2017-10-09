import cv2
import numpy as np

class PerspectiveTransformer:
    def __init__(self):
        self.M = None           # Transformation Matrix
        self.Minv = None        # Inverse Matrix of M
        self.srcPoints = None
        self.dstPoints = None
        self._calculate_matrix()

    def _calculate_matrix(self):
        filename = 'test_images/straight_lines1.jpg'
        img = cv2.imread(filename)
        img_size = (img.shape[1], img.shape[0])
        self.srcPoints = np.float32(
                        [[img_size[0] * 0.151, img_size[1]],
                        [img_size[0] * 0.451, img_size[1] * 0.64],
                        [img_size[0] * 0.553, img_size[1] * 0.64],
                        [img_size[0] * 0.888, img_size[1]]])

        self.dstPoints = np.float32(
                        [[(img_size[0] / 4) - 30, img_size[1]],
                        [(img_size[0] / 4 - 30), 0],
                        [(img_size[0] * 3 / 4 + 30), 0],
                        [(img_size[0] * 3 / 4 + 30), img_size[1]]])

        self.M = cv2.getPerspectiveTransform(self.srcPoints, self.dstPoints)
        self.Minv = cv2.getPerspectiveTransform(self.dstPoints, self.srcPoints)

    def warp_perspective(self, img):
        if self.M is None:
            self._calculate_matrix()
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)
