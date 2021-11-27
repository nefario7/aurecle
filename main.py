# Main
# * Dependencies
import argparse
import os
import cv2 as cv

# * Imports
from slic import SlicSegmentation
from utils import *
from processing import lab_segmentation

base_path = r"sample"

parser = argparse.ArgumentParser(description="Aurecle Processing")
parser.add_argument("--image", dest="image", required=True, type=str, help="image name in sample")
parser.add_argument("--process", dest="process", required=True, type=str, help="process to run on the image")
args = parser.parse_args()


if __name__ == "__main__":
    if args.process == "seg":
        print("Running Segmentation...")
        slic = SlicSegmentation(k=200, m=20)
        slic.process(args.image)

    if args.process == "lab":
        print("Running LAB Segmentation...")

        if args.process.split("_")[-1] == "seg":
            slic = SlicSegmentation(k=200, m=20)
            slic_image = slic.process(args.image)
        else:
            slic_image = cv.imread(os.path.join(base_path, args.image))
        segmented_lab = lab_segmentation(slic_image, 0, 106, 105, 164, 51, 177)
        save(segmented_lab, args.image, "lab")
        show(segmented_lab)
