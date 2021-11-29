import cv2
import numpy as np 

#for parameter tuning GUI
def callback(x):
	pass

if __name__ == "__main__":

    #GUI
    cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp',600,600)
    cv2.createTrackbar('threshold_value','disp',0,100,callback)

    image_org = cv2.imread('images/fort-pitt-google.jpeg')
    cv2.imshow('Original', image_org)
    image = cv2.imread('images/slic-seg-fortpitt.png', 0)
    image_copy = image.copy()
    image_color  = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_color_copy = image_color.copy()

    while(1):
        threshold_value = cv2.getTrackbarPos('threshold_value','disp')*0.01
        print(threshold_value)

        #remove grey regions
        for row in image:
            if (sum(row) < threshold_value* 128 * image.shape[1]):
                row[:] = 0

        #detect contours
        cont, hier = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(cont, key = cv2.contourArea, reverse=True)
        cv2.drawContours(image_color, sorted_contours, 0, (255,0,255), 2, cv2.LINE_AA)
        rect = cv2.boundingRect(sorted_contours[0])
        x, y, w, h = rect
        cv2.rectangle(image_color,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("disp", image)
        cv2.imshow("contour", image_color)
        image = image_copy.copy()
        image_color = image_color_copy.copy()

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.waitKey()
    cv2.destroyAllWindows()
