
import cv2
import numpy as np
import sys 
import os


'''
[29 29 29]
[163 163 163]


[  0 105  51]
[106 164 177]



new images 
[  5 128 114]
[ 33 165 131]
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

filename = "326"
image_name = filename + "_out_m30_k400.png"

image = cv2.imread(image_name)

lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.namedWindow("Output")





segmented_lab = Lab_Segmentation(lab_image, 5, 33, 128, 165, 114, 131)
gray = cv2.cvtColor(segmented_lab, cv2.COLOR_BGR2GRAY)
cv2.imwrite(filename + 'image_threshold.png',gray)
cv2.imshow("Output", gray)

cv2.waitKey(0)

cv2.destroyAllWindows()



