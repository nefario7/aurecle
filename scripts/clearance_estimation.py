import cv2
import numpy as np
import operator

class HeightEstimation(object):
    def __init__(self, frame_org, frame_bbox, bbox_params): 
        self.frame_org = frame_org
        self.frame_bbox = frame_bbox
        self.bbox_params = bbox_params
        self.pick = [(179, 459), (439, 326), (591, 322), (761, 434)]
        self.LANE_WIDTH = 12
        self.height = None

    def get_height(self):
        canny_image  = self.__canny_edge_detector()
        masked_image = self.__region_of_interest(canny_image) 
        lines = self.__houghTransform(masked_image)
        lines_coord = self.__average_slope_intercept(lines) 
        line_image = self.__display_lines(lines_coord)
        line_overlay_image = self.__comb_image(line_image)
        inter0, inter1, lane_width_pix, inter_overlay_image = self.__lane_width_pix(lines_coord, line_overlay_image)
        height_overlay_image = self.__height_estimate(inter0, inter1, lane_width_pix, inter_overlay_image)
        return height_overlay_image 

    def __canny_edge_detector(self):
        lower_color_bound_white  = np.array([152, 114, 63])  
        upper_color_bound_white  = np.array([255, 255, 255])  
        lower_color_bound_yellow = np.array([71,  81,  80]) 
        upper_color_bound_yellow = np.array([105, 157, 184]) 
        mask1  = cv2.inRange(self.frame_bbox, lower_color_bound_white,  upper_color_bound_white)
        mask2  = cv2.inRange(self.frame_bbox, lower_color_bound_yellow, upper_color_bound_yellow)
        mask   = cv2.bitwise_or(mask1, mask2)
        target = cv2.bitwise_and(self.frame_bbox, self.frame_bbox, mask=mask)
        canny  = cv2.Canny(target, 69, 200) 
        return canny

    def __region_of_interest(self, image):
        polygon= np.asarray(self.pick)
        mask = np.zeros_like(image)
        cv2.fillConvexPoly(mask, polygon, 255) 
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
    
    def __houghTransform(self, image):
        lines = cv2.HoughLinesP(image, 2, np.pi/180, 100,
                                np.array([]), minLineLength = 40,
                                maxLineGap = 5) 
        return lines

    def __average_slope_intercept(self, lines):
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis = 0)
        right_fit_average = np.average(right_fit, axis = 0)
        left_line = self.__create_coordinates(self.frame_org, left_fit_average)
        right_line = self.__create_coordinates(self.frame_org, right_fit_average)
        return np.array([left_line, right_line])

    def __create_coordinates(self, image, line_parameters):
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (3 / 5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def __display_lines(self, lines):
        line_image = np.zeros_like(self.frame_org)
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return line_image

    def __comb_image(self, image):
        line_overlay_image = cv2.addWeighted(self.frame_org, 0.8, image, 1, 1)
        return line_overlay_image

    def __lane_width_pix(self, lines_coord, image):
        inter0 = tuple(self.__bbox_intersection(lines_coord[0]).astype(int))
        inter1 = tuple(self.__bbox_intersection(lines_coord[1]).astype(int))
        r = 5
        c = [0, 0, 255]
        t = 5
        inter_overlay_image = cv2.circle(image, inter0, r, c, t)
        inter_overlay_image = cv2.circle(image, inter1, r, c, t)
        lane_width_pix = int(np.linalg.norm(np.asarray(inter0) - np.asarray(inter1)))
        return inter0, inter1, lane_width_pix, inter_overlay_image

    def __bbox_intersection(self, line):
        __, y, __, h = self.bbox_params
        parameters = np.polyfit((line[0], line[2]), (line[1], line[3]), 1)
        slope = parameters[0]
        intercept = parameters[1]
        y_int = y + h
        x_int = (y_int - intercept)/slope
        return np.array([x_int, y_int])

    def __height_estimate(self, inter0, inter1, lane_width_pix, image):
        __, __, __, height_pix = self.bbox_params
        self.height = self.LANE_WIDTH*(height_pix/lane_width_pix)
        height_overlay_image = cv2.line(image, tuple(map(operator.add, inter1, (-1*int(lane_width_pix/2), 0))), \
                                        tuple(map(operator.add, inter1, (-1*int(lane_width_pix/2), -1*height_pix))), (0, 255, 0), thickness=2)
        height_overlay_image = cv2.line(image, inter0, inter1, (255, 0, 0), thickness=2)
        height_overlay_image = cv2.line(image, tuple(map(operator.add, inter0, (0, -1*height_pix))), tuple(map(operator.add, inter1, (0, -1*height_pix))), (255, 0, 0), thickness=2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (480, 450)
        fontScale = 1
        color = [0, 255, 0]
        thickness = 2
        height_overlay_image = cv2.putText(image, str(round(self.height,2))+str('ft'), org, font, fontScale, color, thickness, cv2.LINE_AA)
        return height_overlay_image
