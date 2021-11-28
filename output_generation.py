import cv2 
import numpy as np 


image = cv2.imread('segmentation-images-test/fort-pitt.jpg', 0)
thresh = cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

cv2.imshow("output", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()