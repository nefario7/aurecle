import math
import os
import numpy as np
import string
import random
import cv2 as cv
from skimage import io

base_path = r"sample"


def show(image):
    letters = string.ascii_lowercase
    name = "".join(random.choice(letters) for i in range(10))
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def save(image, name, mode="cv", pref="", ext=""):
    save_path = os.path.join(base_path, pref + name.split(".")[0] + ext + ".png")
    if mode == "cv":
        cv.imwrite(save_path, image)
    elif mode == "sk":
        io.imsave(save_path, image)
    print("Image saved at : ", save_path)
