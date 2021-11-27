import cv2 as cv
import os
import numpy as np
from tqdm import tqdm

base_path = r"pitt_gray1_out_m20_k200.png"


def lab_segmentation(image, L_lower, L_upper, a_lower, a_upper, b_lower, b_upper):
    image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    lowerRange = np.array([L_lower, a_lower, b_lower], dtype="uint8")
    upperRange = np.array([L_upper, a_upper, b_upper], dtype="uint8")
    mask = image[:].copy()
    # imageLab = cv.cvtColor(image, cv.COLOR_BGR2Lab)
    imageRange = cv.inRange(image, lowerRange, upperRange)

    mask[:, :, 0] = imageRange
    mask[:, :, 1] = imageRange
    mask[:, :, 2] = imageRange

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    faceLab = cv.bitwise_and(image, mask)

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    return faceLab


# class Segmentation:
#     def __init__(self, path):
#         self.path = path
#         self.image = None
#         self.hsv_image = None

#     def read_image(self):
#         self.image = cv.imread(self.path)

#     def process_image(image, type):
#         gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#         # * Gradients
#         if type == "gradients":
#             gradientsx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=5)
#             gradientsy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=5)
#             return [gradientsx, gradientsy]

#         # * Binarization
#         if type == "binary":
#             _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#             return binary

#         # * Convert to HSV
#         if type == "hsv":
#             return cv.cvtColor(image, cv.COLOR_BGR2HSV)

#     def combine(self, show=False):
#         # * Combine the superpixels into larger segments

#         # * Show the segments
#         if show:
#             self.show(self.hsv_image[:, :, 2])
#         pass


# if __name__ == "__main__":

#     segmentation = Segmentation(base_path)
#     segmentation.read_image()
#     processed_image = segmentation.process_image()
#     segmentation.save(processed_image[0], base_path.split("_")[0] + "X")
#     segmentation.save(processed_image[1], base_path.split("_")[0] + "Y")
#     # segmentation.show(processed_image)
