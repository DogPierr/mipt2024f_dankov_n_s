import cv2
import numpy as np


class Localizator:
    def __init__(self):
        """
        Initializes the Localizator class.

        Creates BarcodeDetector and QRCodeDetector instances.
        """
        self.bd = cv2.barcode.BarcodeDetector()
        self.qcd = cv2.QRCodeDetector()

    def localize(self, img) -> list[int]:
        """
        Localizes all barcodes and QR codes in the given image.

        :param img: Image where barcodes and QR codes should be localized.
        :return: List of 2D points of the detected barcodes and QR codes.
        """
        barcodes = []

        bd_ret, _, bd_points, _ = self.bd.detectAndDecodeMulti(img)
        qcd_ret, _, qcd_points, _ = self.qcd.detectAndDecodeMulti(img)

        if bd_ret:
            barcodes.extend(np.array(bd_points, dtype=np.int32))

        if qcd_ret:
            barcodes.extend(np.array(qcd_points, dtype=np.int32))

        return barcodes
