import cv2 as cv
import os
import numpy as np
from tqdm import tqdm

base_path = r"6-out_m20_k200.png"

class Segmentation():
    def __init__(self, path):
        self.path=path
        self.image = None
        self.hsv_image = None
        pass
    def read_img(self):
        self.image = cv.imread(self.path)

    def slic(self):
        pass

    def convert_to_hsv(self):
        self.hsv_image = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)


    def combine(self, show = False):
        # * Combine the superpixels into larger segments
        

        # * Show the segments
        if show:
            self.show(self.hsv_image[:,:,2])
        pass 
    
    def show(self, image):
        cv.imshow("Rando_str", image )
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__=="__main__":

    segmentation = Segmentation(base_path)
    segmentation.read_img()
    segmentation.convert_to_hsv()
    segmentation.combine(show = True)





