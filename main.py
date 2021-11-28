# Main
# * Dependencies
import argparse
import os
import cv2 as cv

# * Imports
from slic import SlicSegmentation
from utils import *
from processing import *

base_path = r"sample"

parser = argparse.ArgumentParser(description="Aurecle Processing")
parser.add_argument("--image", dest="image", required=False, type=str, help="image name in sample")
parser.add_argument("--process", dest="process", required=False, type=str, help="process to run on the image")
parser.add_argument("--video", dest="video", required=False, type=str, help="video name to process")
args = parser.parse_args()

"""
Commands:
python main.py --video "sample_video\13_19_49.mp4"
python main.py --image "sample\sample_c.png" --process "aurecle"
python main.py --image "sample_grabs\sample_c.png" --process "slic"
"""

if __name__ == "__main__":
    if args.video is not None:
        video_processing(args.video)

    if args.image is not None:
        # * Complete Aurecle Pipeline
        if args.process == "aurecle":
            print("Running Aurecle Pipeline...")
            rimg = cv.imread(args.image)
            _, _, _ = aurecle_segmentation(rimg, m=30, k=400)

        # * SLIC Segmentation
        elif args.process == "slic":
            print("Running SLIC Segmentation...")
            rimg = cv.imread(args.image)
            slic = SlicSegmentation(m=20, k=200)
            slic_image = slic.process(rimg)
            save(slic_image, "temp", mode="sk")

            # slic_image = slic_image[:, :, ::-1]
            # save(slic_image, "temp_bgr2rgb", mode="cv")
