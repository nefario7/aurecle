# Main
# * Dependencies
import argparse
import os
import cv2 as cv

# * Imports
from slic import SlicSegmentation
from utils import *
from processing import lab_segmentation, video_processing

base_path = r"sample"

parser = argparse.ArgumentParser(description="Aurecle Processing")
parser.add_argument("--image", dest="image", required=False, type=str, help="image name in sample")
parser.add_argument("--process", dest="process", required=False, type=str, help="process to run on the image")
parser.add_argument("--video", dest="video", required=False, type=str, help="video name to process")
args = parser.parse_args()

"""
Commands:
python main.py --video "sample_video\13_19_49.mp4"
"""

if __name__ == "__main__":
    if os.path.isfile(args.video):
        video_processing(args.video)

    if args.process == "seg":
        print("Running Segmentation...")
        slic = SlicSegmentation(k=200, m=20)  # range of m = [1,40]
        slic.process(args.image)

    if args.process == "lab":
        print("Running LAB Segmentation...")

        if args.process.split("_")[-1] == "seg":
            slic = SlicSegmentation(k=200, m=20)
            slic_image = slic.process(args.image)
        else:
            slic_image = cv.imread(os.path.join(base_path, args.image))
        segmented_lab = lab_segmentation(slic_image, 0, 106, 105, 164, 51, 177)
        save(segmented_lab, args.image, ext="_lab")
        show(segmented_lab)
