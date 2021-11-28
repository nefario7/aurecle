import math
import os
import numpy as np
import string
import random
import cv2 as cv
from skimage import io

base_path = r"sample"


def show(image, header = None):
    if header is None:
        letters = string.ascii_lowercase
        header = "".join(random.choice(letters) for i in range(10))
    cv.namedWindow(header, cv.WINDOW_NORMAL)
    cv.imshow(header, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def save(image, name, mode="cv", pref="", ext=""):
    save_path = os.path.join(base_path, pref + name.split(".")[0] + ext + ".png")
    if mode == "cv":
        cv.imwrite(save_path, image)
    elif mode == "sk":
        io.imsave(save_path, image)
    print("Image saved at : ", save_path)


def bbox_intersection(self, y, h, line):
    slope = (line[3] - line[1]) / (line[2] - line[0])
    y = y + h
    x = ((y - line[1]) / slope) + line[0]

    return np.array([x, y])
