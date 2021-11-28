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
python main.py --image "sample\pitt.jpeg" --process "aurecle"
"""

if __name__ == "__main__":
    if args.video is not None:
        video_processing(args.video)

    if args.image is not None:
        if args.process == "aurecle":
            print("Running Aurecle Pipeline...")
            rimg = cv.imread(args.image)
            _, _, _ = aurecle_segmentation(rimg, m=30, k=400)
            # slic = SlicSegmentation(m=30, k=400)
            # slic_image = slic.process(rimg)
            # show(slic_image)
        elif args.process == "slic":
            print("Running SLIC...")
            rimg = cv.imread(args.image)
            rimg = cv.cvtColor(rimg, cv.COLOR_BGR2RGB)
            slic = SlicSegmentation(m=20, k=200)
            slic_image = slic.process(rimg)

            save(slic_image, "temp", mode="sk")
            slic_image = cv.cvtColor(slic_image, cv.COLOR_BGR2RGB)
            save(slic_image, "temp_bgr2rgb", mode="cv")

    # if args.process == "lab":
    #     print("Running LAB Segmentation...")

    #     if args.process.split("_")[-1] == "seg":
    #         slic = SlicSegmentation(k=200, m=20)
    #         slic_image = slic.process(args.image)
    #     else:
    #         slic_image = cv.imread(os.path.join(base_path, args.image))
    #     segmented_lab = lab_segmentation(slic_image, 0, 106, 105, 164, 51, 177)
    #     save(segmented_lab, args.image, ext="_lab")
    #     show(segmented_lab)
