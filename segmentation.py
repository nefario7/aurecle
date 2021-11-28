import cv2
import numpy as np
from slic import SlicSegmentation
from utils import *
from processing import *


class Segmentation():
    def __init__(self, frame):
        self.image = frame
        self.rect = None
    def segment(self):
        '''
        input: frame 
        output: bbox for the estimation class 

        Steps:
        Get the frame 
        Do the segmentation 
        Do LAB Thresholding 
        Contour Detection and post processing
        Bbox pass to clearance estimation class 
        '''
        slic = SlicSegmentation(m= 40, k = 400)
        segmented_image = slic.process(self.image)
        lab_threshold = lab_segmentation(segmented_image, 5, 33, 128, 165, 114, 131)
        lab_threshold = cv2.cvtColor(lab_threshold, cv2.COLOR_BGR2GRAY)
        self.rect = contour_process(lab_threshold)
        return self.rect

