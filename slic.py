import math
from skimage import io, color
from skimage.transform import resize
import numpy as np
from tqdm import tqdm

from utils import *

base_path = r"segmentation-images-test/"
output_path = r"outputs"

# function for the Simple Linear Iterative Clustering
def slic(S, img, img_h, img_w, clusters, tag, dis, imgname):
    clusters = initial_cluster_center(S, img, img_h, img_w, clusters)
    reassign_cluster_center_acc_to_grad(clusters, img)
    for i in range(10):  # usually the algortihm converges within 10 iterations
        assign_pixels_to_cluster(clusters, S, img, img_h, img_w, tag, dis)
        update_cluster_mean(clusters)
        if i == 9:  # to print the output after 10 iterations
            name = "{imgname}_out_m{m}_k{k}.png".format(loop=i, m=m, k=k)
            avg_color_cluster(img, name, clusters, imgname)
    return clusters


if __name__ == "__main__":

    for i in tqdm(range(1, 9)):
        # read the input RGB image
        image_path = base_path + str(i) + ".jpeg"
        rgb = io.imread(image_path, plugin="matplotlib")
        print(rgb.shape)

        # input images are resized to (400 x 400) for processing
        img = resize(rgb, (400, 400), anti_aliasing=True)
        print(img.shape)

        # convert RGB to LAB
        img = color.rgb2lab(img)

        k = 200  # Number of Super pixels
        m = 20  # Constant for normalizing the color proximity, range of m = [1,40]

        img_h = img.shape[0]  # Image Height
        img_w = img.shape[1]  # Image Width

        N = img_h * img_w  # Total number of pixels in the image
        S = int(math.sqrt(N / k))  # average size of each superpixel

        clusters = []
        tag = {}
        # initialize the distance between pixels and cluster center as infinity
        dis = np.full((img_h, img_w), np.inf)
        cluster = slic(S, img, img_h, img_w, clusters, tag, dis, str(i))

    # superpixels
    # for c in cluster:
    #     print("H {} : W {}, l {}, a {}, b {}".format(c.h, c.w, c.l, c.a, c.b))
