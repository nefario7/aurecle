# Import the required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edge_detector(image):
    flag = 0
    if (flag==0):
        lower_color_bound_white = np.array([152, 114, 63])  
        upper_color_bound_white = np.array([255, 255, 255])  
        lower_color_bound_yellow = np.array([71, 81, 80]) 
        upper_color_bound_yellow = np.array([105, 157, 184]) 
        mask1 = cv2.inRange(image, lower_color_bound_white, upper_color_bound_white)
        mask2 = cv2.inRange(image, lower_color_bound_yellow, upper_color_bound_yellow)
        mask = cv2.bitwise_or(mask1, mask2)
        cv2.imshow('mask', mask)
        cv2.imwrite('clearance_estimation_pipeline/color_thres_mask.jpg', mask)
        target = cv2.bitwise_and(image,image, mask=mask)
        cv2.imshow('target', target)
        cv2.imwrite('clearance_estimation_pipeline/color_thres_mask_overlay.jpg', target)
        canny = cv2.Canny(target, 69, 200) #69, 200
        cv2.imwrite('clearance_estimation_pipeline/canny.jpg', canny)
    elif (flag==1):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
        canny = cv2.Canny(blur, 69, 200) #69, 200
    return canny

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        mousePick(x, y)

def mousePick(x, y):
    global pick, frame
    frame_copy = frame.copy()
    pick.append((x,y))
    for i in range(len(pick)):
        frame_copy = cv2.circle(frame_copy, pick[i], 5, (0, 0, 255), 2)
        frame_copy = cv2.putText(frame_copy, str(i), (pick[i][0]+10, pick[i][1]-10),\
                          cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 1)
    cv2.imshow("img", frame_copy)
    cv2.imwrite('clearance_estimation_pipeline/frame.jpg', frame)
    cv2.waitKey(1)
    if len(pick) > 3:
        print('Points selected: ', pick)
        print('Point 0 BGR: ', frame[tuple(reversed(pick[0]))])
        print('Point 1 BGR: ', frame[tuple(reversed(pick[1]))])
        print('Point 2 BGR: ', frame[tuple(reversed(pick[2]))])
        print('Point 3 BGR: ', frame[tuple(reversed(pick[3]))])
        pick =  [(179, 459), (439, 326), (591, 322), (761, 434)] #works on 334 and 304
        #pick = [(253, 446), (346, 387), (683, 374), (782, 431)] #works on 334
        region_of_interest()

def region_of_interest():
    global pick, frame, canny_image
    polygon= np.asarray(pick)
    mask = np.zeros_like(canny_image)
    cv2.fillConvexPoly(mask, polygon, 255) 
    masked_image = cv2.bitwise_and(canny_image, mask)
    cv2.imshow("masked image", masked_image)
    cv2.imwrite('clearance_estimation_pipeline/region_of_interest.jpg', masked_image)
    houghTransform(masked_image) 

def houghTransform(image):
    lines = cv2.HoughLinesP(image, 2, np.pi / 180, 100,
							np.array([]), minLineLength = 40,
							maxLineGap = 5) 
    print(len(lines))
    average_slope_intercept(lines) 

def average_slope_intercept(lines):
    global frame
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        # It will fit the polynomial and the intercept and slope
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = create_coordinates(frame, left_fit_average)
    right_line = create_coordinates(frame, right_fit_average)
    display_lines(np.array([left_line, right_line]))

def create_coordinates(image, line_parameters):
	slope, intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(y1 * (3 / 5))
	x1 = int((y1 - intercept) / slope)
	x2 = int((y2 - intercept) / slope)
	return np.array([x1, y1, x2, y2])

def display_lines(lines):
    global frame
    line_image = np.zeros_like(frame)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    comb_image(line_image)

def comb_image(line_image):
    global frame
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("results", combo_image)
    cv2.imwrite('clearance_estimation_pipeline/lane_detection.jpg', combo_image)

if __name__ == '__main__':
	
    frame = cv2.imread('inputs/320.jpg') #304 #325 #328 #329 #331 #333 #334
    scale_percent = 50 
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim)
    print(frame.shape)
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.imshow("img", frame)

    canny_image = canny_edge_detector(frame)
    cv2.imshow("canny image", canny_image)

    print("Select 4 points")
    pick = []
    cv2.setMouseCallback("img", click)

    cv2.waitKey(0)
    cv2.destroyAllWindows()