
import cv2
import numpy as np
import sys 
import os


'''
[29 29 29]
[163 163 163]


[  0 105  51]
[106 164 177]
'''

def Lab_Segmentation(image,L_lower, L_upper, a_lower, a_upper, b_lower, b_upper):
    lowerRange= np.array([L_lower, a_lower, b_lower] , dtype="uint8")
    upperRange= np.array([L_upper, a_upper, b_upper], dtype="uint8")
    mask = image[:].copy()
    # imageLab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    imageRange = cv2.inRange(image,lowerRange, upperRange)
    
    mask[:,:,0] = imageRange
    mask[:,:,1] = imageRange
    mask[:,:,2] = imageRange
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    faceLab = cv2.bitwise_and(image,mask)

    return faceLab





image = cv2.imread('outputs/pitt_out_m20_k200.png')

lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.namedWindow("Output")





segmented_lab = Lab_Segmentation(lab_image, 0, 106, 105, 164, 51, 177)
gray = cv2.cvtColor(segmented_lab, cv2.COLOR_BGR2GRAY)
cv2.imwrite('lab_image_threshold.png',gray)
cv2.imshow("Output", gray)

cv2.waitKey(0)

cv2.destroyAllWindows()



