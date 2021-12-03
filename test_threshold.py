import cv2 as cv
from skimage.transform import resize
from utils import *
from processing import *
'''
[  0 106 114]
[ 99 168 163]
'''
def crop_image(image):
    crop_mask = np.zeros_like(image)
    CROP_X = 100
    CROP_W = 200
    CROP_Y = 100
    CROP_H = 200
    crop_mask[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X + CROP_W]  = np.ones((CROP_H, CROP_W))
    frame = cv.bitwise_and(image, image, mask  = crop_mask)
    return frame

image = cv.imread('segmented-image.png')

thresh = lab_segmentation(image,0, 99, 106, 168, 114, 163 )
thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
thresh = crop_image(thresh)
show(thresh, "thresh")