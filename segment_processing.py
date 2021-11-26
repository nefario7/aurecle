import cv2 as cv
import os
import numpy as np
from tqdm import tqdm
import string
import random



base_path = r"pitt_gray1_out_m20_k200.png"


class Segmentation:
    def __init__(self, path):
        self.path = path
        self.image = None
        self.hsv_image = None

    def read_image(self):
        self.image = cv.imread(self.path)

    def process_image(image, type):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # * Gradients
        if type == "gradients":
            gradientsx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=5)
            gradientsy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=5)
            return [gradientsx, gradientsy]

        # * Binarization
        if type == "binary":
            _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            return binary

        # * Convert to HSV
        if type == "hsv":
            return cv.cvtColor(image, cv.COLOR_BGR2HSV)

    def slic(self):
        pass

    def combine(self, show=False):
        # * Combine the superpixels into larger segments

        # * Show the segments
        if show:
            self.show(self.hsv_image[:, :, 2])
        pass

    def show(self, image):
        letters = string.ascii_lowercase
        name = "".join(random.choice(letters) for i in range(10))
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.imshow(name, image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def save(self, image, name):
        cv.imwrite(name + "save.jpeg", image)


if __name__ == "__main__":

    segmentation = Segmentation(base_path)
    segmentation.read_image()
    processed_image = segmentation.process_image()
    segmentation.save(processed_image[0], base_path.split("_")[0] + "X")
    segmentation.save(processed_image[1], base_path.split("_")[0] + "Y")
    # segmentation.show(processed_image)
