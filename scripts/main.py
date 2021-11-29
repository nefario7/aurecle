import cv2
import numpy as np
from clearance_estimation import HeightEstimation

frame = cv2.imread('images/334.jpg')
scale_percent = 50 
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
frame_org = cv2.resize(frame, dim)  
bbox_params = [444, 170, 311, 160] #[x,y,w,h]
frame_bbox = frame_org.copy()
frame_bbox = cv2.rectangle(frame_bbox, (bbox_params[0], bbox_params[1]), (bbox_params[0]+bbox_params[2], bbox_params[1]+bbox_params[3]), (255,0,0), 2)
cv2.imshow('INPUT', frame_org)

frame_est = HeightEstimation(frame_org, frame_bbox, bbox_params)
height_overlay_image = frame_est.get_height()
height = frame_est.height

cv2.imshow('OUTPUT', height_overlay_image)
print('Clearance height is: ', round(height,2))
        
cv2.waitKey(0)
cv2.destroyAllWindows()