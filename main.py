# Main
# * Dependencies
import argparse
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from clearance_estimation import HeightEstimation

# * Imports
from slic import SlicSegmentation
from utils import *
from processing import *


# Argument Parser
parser = argparse.ArgumentParser(description="Aurecle Processing")
parser.add_argument("--image", dest="image", required=False, type=str, help="image name in sample")
parser.add_argument("--process", dest="process", required=False, type=str, help="process to run on the image")
parser.add_argument("--video", dest="video", required=False, type=str, help="video name to process")
parser.add_argument("--directory", dest="dir", required=False, type=str, help="path of directory to process frames")

args = parser.parse_args()

if __name__ == "__main__":
    # * Run Segmenation Pipeline
    if args.process == "segmentation":
        dir_path = args.dir
        directory_process(dir_path)

    # * Run Height Estimation Pipeline
    if args.process == "heightestimation":
        height_estimated_hist = []
        height_estimated_moving_avg = []
        height_estimated_moving_avg_avg = []
        bbox_params_hist = np.load('processing/input_to_main/bbox.npy')
        #frame_count = len(bbox_params_hist) 
        frame_count = 68

        for i in range(frame_count):
            filename = str(i+1) + str('.jpg')
            filepath = str('processing/input_to_main/') + filename
            frame = cv.imread(filepath)
            print(filepath)

            scale_percent = 50 
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            frame_org = cv.resize(frame, dim)  

            bbox_params = bbox_params_hist[i][1:5] #[x,y,w,h]
            bbox_params[0] = int((bbox_params[0]/400)*960) 
            bbox_params[1] = int((bbox_params[1]/400)*540) 
            bbox_params[2] = int((bbox_params[2]/400)*960) 
            bbox_params[3] = int((bbox_params[3]/400)*540) 
            #bbox_params = [444, 170, 311, 160] 

            frame_bbox = frame_org.copy()
            frame_bbox = cv.rectangle(frame_bbox, (bbox_params[0], bbox_params[1]), (bbox_params[0]+bbox_params[2], bbox_params[1]+bbox_params[3]), (255,0,0), 2)
            frame_est = HeightEstimation(frame_org, frame_bbox, bbox_params, np.mean(height_estimated_moving_avg))
            height_overlay_image = frame_est.get_height()
            height_estimated = frame_est.height
            height_estimated_hist.append(height_estimated)
            height_estimated_moving_avg.append(np.mean(height_estimated_hist))
            height_estimated_moving_avg_avg.append(np.mean(height_estimated_moving_avg))

            cv.imwrite('processing/output_from_main/' + filename.rsplit(".", 1)[0] + '-overlay.jpg', height_overlay_image)
            print('Clearance height is: ', round(height_estimated,2))

        x = np.arange(1,69) 
        ground_truth = np.ones(len(x))*13.5

        # Height Estimation Plot
        plt.title("Height Estimate by Aurecle", fontsize=20) 
        plt.xlabel("Frame", fontsize=15)
        plt.ylabel("Estimated Height $[ft]$", fontsize=15) 
        plt.plot(x, height_estimated_hist, label='Raw estimate') 
        plt.plot(x, height_estimated_moving_avg, label='Filtered estimate') 
        plt.plot(x, ground_truth, label='Ground truth')
        plt.legend(fontsize=13)
        plt.show()

        cv.waitKey(0)
        cv.destroyAllWindows()

    # * Process just a video (TESTING ONLY)
    # if args.video is not None:
    #     video_processing(args.video)

    # * Processing segmentation on images (TESTING ONLY)
    # if args.image is not None:
    #     # * Complete Aurecle Pipeline
    #     if args.process == "aurecle":
    #         print("Running Aurecle Pipeline...")
    #         rimg = cv.imread(args.image)
    #         _, _, _ = aurecle_segmentation(rimg, m=20, k=200)

    #     # * SLIC Segmentation
    #     elif args.process == "slic":
    #         print("Running SLIC Segmentation...")
    #         rimg = cv.imread(args.image)
    #         slic = SlicSegmentation(m=20, k=200)
    #         slic_image = slic.process(rimg)
    #         save(slic_image, "temp", mode="sk")
