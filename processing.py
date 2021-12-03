import cv2 as cv
import os
import numpy as np
from tqdm import tqdm
import glob
# * Imports
from skimage.util import img_as_ubyte
from slic import SlicSegmentation
from utils import *

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

def crop_image(image):
    crop_mask = np.zeros_like(image)
    CROP_X = 175
    CROP_W = 120
    CROP_Y = 100
    CROP_H = 200
    crop_mask[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X + CROP_W]  = np.ones((CROP_H, CROP_W))
    frame = cv.bitwise_and(image, image, mask  = crop_mask)
    return frame

def contour_process(image):
    threshold_value = 0.02
    image_copy = image.copy()
    # image[350:400, :] = 0

    for row in image:
        if sum(row) < threshold_value * 55 * image.shape[1]:
            row[:] = 0

    cont, hier = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if(len(cont)<1):
        return None
    sorted_contours = sorted(cont, key=cv.contourArea, reverse=True)
    
    # cv.drawContours(image_color, sorted_contours, 0, (255,0,255), 2, cv.LINE_AA)
    rect = cv.boundingRect(sorted_contours[0])
    return rect

def aurecle_segmentation(image, m=40, k=400):
    slic = SlicSegmentation(m, k)

    segmented_image = slic.process(image)
    # show(segmented_image, "segmented image")
    segmented_image = img_as_ubyte(segmented_image)    
    save(segmented_image, 'segmented_image')
    lab_threshold = lab_segmentation(segmented_image,0, 99, 106, 168, 114, 163 )
    save(lab_threshold, 'lab_threshold')
    lab_threshold = cv.cvtColor(lab_threshold, cv.COLOR_BGR2GRAY)
    lab_threshold = crop_image(lab_threshold)

    # lab_threshold =cv.resize(lab_threshold, (960,540))
    x, y, w, h = contour_process(lab_threshold)
    
    frame_rect = [x,y,w,h]

    bbox_frame = cv.rectangle(
                lab_threshold, (frame_rect[0], frame_rect[1]), (frame_rect[0] + frame_rect[2], frame_rect[1] + frame_rect[3]), (255, 0, 0), 2
            )
    save(bbox_frame,"bbox")
    return [x, y, w, h], segmented_image, lab_threshold


def video_processing(video_path):
    output_path = os.path.join(video_path.split(".")[0] + "_output.avi")
    print("Video output path : ", output_path)

    input_video = cv.VideoCapture(video_path)
    output_video = cv.VideoWriter(output_path, cv.VideoWriter_fourcc("M", "J", "P", "G"), 1, (400, 400))

    # input_video = video.Video(video_path)
    # print(input_video.frame_count(), input_video.duration())
    # output_video = cv.VideoWriter(output_path, cv.VideoWriter_fourcc("M", "J", "P", "G"), 1, (400, 400))

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

def directory_process(dir_path):
    # output_path = os.path.join(dir_path , "output.avi")
    # print("Video output path : ", output_path)


    # fourcc = cv.VideoWriter_fourcc("MJPG")
    # writer = None
    # (h, w) = (400,400)

    # zeros = None
    print(dir_path)

    bboxs = []
    for i in range(len(os.listdir(dir_path))):
        frame_no = i+1
        print("Processing Frame no: " + str(frame_no))
        frame = cv.imread(dir_path+"/"+str(frame_no)+ ".jpg")
        [x,y,w,h], segmented_image, lab_threshold = aurecle_segmentation(frame, m = 20, k=200)
        bbox = [frame_no, x, y, w, h]
        bboxs.append(bbox)
        np.save('bbox.npy', bboxs)
        print("saved for frame: "+ str(frame_no))
        frame_rect = [x,y,w,h]

        bbox_frame = cv.rectangle(
            segmented_image, (frame_rect[0], frame_rect[1]), (frame_rect[0] + frame_rect[2], frame_rect[1] + frame_rect[3]), (255, 0, 0), 2
            )
        save(bbox_frame, "bbox_"+str(frame_no),mode="cv")
        save(lab_threshold, "lab_thresh_"+str(frame_no),mode="cv")
        save(segmented_image, "segmented_"+str(frame_no),mode="cv")

        # cv.imshow("output", bbox_frame)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

            
        #! give the output of the frame 
    	# if writer is None:
        #     # store the image dimensions, initialize the video writer,
        #     # and construct the zeros array
        #     (h, w) = frame.shape[:2]
        #     writer = cv.VideoWriter(output_path, fourcc, 1,(w, h), True)

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
