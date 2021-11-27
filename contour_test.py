# %%
import cv2
import numpy as np 

threshold_value = 0.3
image = cv2.imread('lab_image_threshold.png', 0)
image_copy = image.copy()
image_color  = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for row in image:
    if (sum(row) < threshold_value* 128 * image.shape[1]):
        row[:] = 0


cont, hier = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(cont, key = cv2.contourArea, reverse=True)
print(len(cont))
# cv2.drawContours(image_color, sorted_contours, 0, (255,0,255), 2, cv2.LINE_AA)

# %%
rect = cv2.boundingRect(sorted_contours[0])
x, y, w, h = rect

cv2.rectangle(image_color,(x,y),(x+w,y+h),(0,255,0),2)

# %%

M = cv2.moments(sorted_contours[0])

# %%
cv2.imshow('OG', image_copy)
cv2.imshow('output', image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()