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

    def localize_opencv(self, img) -> list[int]:
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

    def localize(self, img, min_thresh: int = 225, max_thresh: int = 255, kernel_size: tuple = (21, 7)) -> list[int]:
        """
        Localizes barcode or QR code in the given image. It should be guranteed that
        there is only one barcode or QR code in the image and it takes the largest part of the image.

        :param img: Image where barcodes and QR codes should be localized.
        :return: List of 2D points of the detected barcodes and QR codes.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gradX = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        gradY = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        blurred = cv2.blur(gradient, (9, 9))
        _, thresh = cv2.threshold(blurred, min_thresh, max_thresh, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Perform erosions and dilations to remove small regions
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)

        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return []

        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        # Compute the rotated bounding box of the largest contour
        hull = cv2.convexHull(c)

        return hull

    def localize_test(
        self, img, min_thresh: int = 225, max_thresh: int = 255, kernel_size: tuple = (21, 7)
    ) -> list[int]:
        """
        Localizes barcode or QR code in the given image. It should be guranteed that
        there is only one barcode or QR code in the image and it takes the largest part of the image.

        :param img: Image where barcodes and QR codes should be localized.
        :return: List of 2D points of the detected barcodes and QR codes.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gradX = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        gradY = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        blurred = cv2.blur(gradient, (9, 9))
        _, thresh = cv2.threshold(blurred, min_thresh, max_thresh, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Perform erosions and dilations to remove small regions
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)

        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return [], gradient, blurred, thresh, closed

        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        # Compute the rotated bounding box of the largest contour
        hull = cv2.convexHull(c)

        return hull, gradient, blurred, thresh, closed
