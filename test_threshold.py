import cv2 as cv
from skimage.transform import resize
from utils import *
from processing import *


image = cv.imread('sample/ret_image_1.png')

thresh = lab_segmentation(image,5, 33, 128, 165, 114, 131 )

show(thresh, "thresh")