import cv2 as cv
import os
import numpy as np
from tqdm import tqdm
import glob

# * Imports
from slic import SlicSegmentation
from utils import bbox_intersection

base_path = r"pitt_gray1_out_m20_k200.png"
rishabh_path = r"/home/rishabh/CV_project/aurecle/sample_video/video_frame/"


def lab_segmentation(image, L_lower, L_upper, a_lower, a_upper, b_lower, b_upper):
    image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    lowerRange = np.array([L_lower, a_lower, b_lower], dtype="uint8")
    upperRange = np.array([L_upper, a_upper, b_upper], dtype="uint8")
    mask = image[:].copy()
    # imageLab = cv.cvtColor(image, cv.COLOR_BGR2Lab)
    imageRange = cv.inRange(image, lowerRange, upperRange)

    mask[:, :, 0] = imageRange
    mask[:, :, 1] = imageRange
    mask[:, :, 2] = imageRange

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    faceLab = cv.bitwise_and(image, mask)

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    return faceLab


def contour_process(image):
    threshold_value = 0.02
    image_copy = image.copy()
    image[300:400, :] = 0

    for row in image:
        if sum(row) < threshold_value * 55 * image.shape[1]:
            row[:] = 0

    cont, hier = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(cont, key=cv.contourArea, reverse=True)
    # cv.drawContours(image_color, sorted_contours, 0, (255,0,255), 2, cv.LINE_AA)
    rect = cv.boundingRect(sorted_contours[0])
    return rect


def aurecle_segmentation(image, m=40, k=400):
    slic = SlicSegmentation(m, k)
    segmented_image = slic.process(image)

    lab_threshold = lab_segmentation(segmented_image, 5, 33, 128, 165, 114, 131)
    lab_threshold = cv.cvtColor(lab_threshold, cv.COLOR_BGR2GRAY)

    x, y, w, h = contour_process(lab_threshold)

    return [x, y, w, h], segmented_image, lab_threshold

def video_processing(video_path):
    input_video = cv.VideoCapture(video_path)
    output_path = os.path.join(video_path.split(".")[0] + "_output.avi")
    print("Video output path : ", output_path)
    output_video = cv.VideoWriter(output_path, cv.VideoWriter_fourcc("M", "J", "P", "G"), 1, (400, 400))

    frame_ctr = 0
    while True:
        ret, frame = input_video.read()
        if frame_ctr % 10 == 0:
            print(f"_________ Processing frame {frame_ctr} _________")
            if ret == False:
                print("Saving Video!")
                break

            frame_rect, seg_img, thr_img = aurecle_segmentation(frame, 30, 400)
            # bbox_y = frame_rect[1]
            # bbox_h = frame_rect[3]

            bbox_frame = cv.rectangle(
                frame, (frame_rect[0], frame_rect[1]), (frame_rect[0] + frame_rect[2], frame_rect[1] + frame_rect[3]), (255, 0, 0), 2
            )
            print("Bounding Box: ", frame_rect)
            cv.imwrite(rishabh_path + str(frame_ctr) + "_bbox" + ".jpg", bbox_frame)
            cv.imwrite(rishabh_path + str(frame_ctr) + "_seg" + ".jpg", seg_img)
            cv.imwrite(rishabh_path + str(frame_ctr) + "_thr" + ".jpg", thr_img)
            output_video.write(bbox_frame)

            # #! Get left and the right lines
            # line_left = None
            # line_right = None

            # intersection_left = bbox_intersection(bbox_y, bbox_h, line_left)
            # intersection_right = bbox_intersection(bbox_y, bbox_h, line_right)

            # #! Stuff to do on the intersection

        frame_ctr += 1

    input_video.release()
    output_video.release()


# class Segmentation:
#     def __init__(self, path):
#         self.path = path
#         self.image = None
#         self.hsv_image = None

#     def read_image(self):
#         self.image = cv.imread(self.path)

#     def process_image(image, type):
#         gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#         # * Gradients
#         if type == "gradients":
#             gradientsx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=5)
#             gradientsy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=5)
#             return [gradientsx, gradientsy]

#         # * Binarization
#         if type == "binary":
#             _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#             return binary

#         # * Convert to HSV
#         if type == "hsv":
#             return cv.cvtColor(image, cv.COLOR_BGR2HSV)

#     def combine(self, show=False):
#         # * Combine the superpixels into larger segments

#         # * Show the segments
#         if show:
#             self.show(self.hsv_image[:, :, 2])
#         pass


# if __name__ == "__main__":

#     segmentation = Segmentation(base_path)
#     segmentation.read_image()
#     processed_image = segmentation.process_image()
#     segmentation.save(processed_image[0], base_path.split("_")[0] + "X")
#     segmentation.save(processed_image[1], base_path.split("_")[0] + "Y")
#     # segmentation.show(processed_image)
