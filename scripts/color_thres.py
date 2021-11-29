import numpy as np
import cv2

# The order of the colors is blue, green, red
lower_color_bounds = np.array([66, 79, 81])
upper_color_bounds = np.array([210, 216, 220])
frame = cv2.imread('images/334.jpg')
scale_percent = 50 
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
frame = cv2.resize(frame, dim)
cv2.imshow("img", frame)

mask = cv2.inRange(frame,lower_color_bounds,upper_color_bounds )
mask_rgb = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
frame = frame & mask_rgb
cv2.imshow('thres_image', frame)

cv2.waitKey()
cv2.destroyAllWindows()